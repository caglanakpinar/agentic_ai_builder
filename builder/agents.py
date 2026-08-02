from abc import abstractmethod
from typing import Any

from builder.prompts import BasePrompt
from builder.tools import ToolBox
from models.llms import BaseLLM
from uilts.configs import AgentConfigs
from uilts.logger import logger
from db_connector.vector import BaseVectorConnector
from db_connector.text import BaseTextConnector
from models.embeddings import BaseEmbeddings


class BaseAgent:
    """Common interface for the agent types a pipeline step can run (judger, thinker, classifier, generator, ...).

    An agent pairs an `AgentConfigs` entry with a live `BaseLLM` caller. Subclasses implement:
      - `_initialize_agent()`: set up any type-specific state (defaults to a no-op).
      - `run(question, context, agent_outputs)`: render this agent's prompt, call the LLM, return the
        agent's output as plain text.

    Prompt rendering is delegated to `BasePrompt`, so an agent's `.md` prompt can reference `{question}`,
    `{context}`, or another agent by name — the latter resolved from `agent_outputs`, which is how one
    step's result feeds the next.

    An agent that declares `tools` gets a `ToolBox` built from them: it imports the function behind each
    tool and renders the schemas in the dialect this agent's provider expects, so the tools passed to a
    call are callable ones. `call_tool` runs whichever of them the model asks for.

    Args:
        name: Agent name as it appears in the YAML `agents:` block, and the key other agents use to
            reference this agent's output.
        agent_config: Parsed config for this agent (prompt directory, tools, mcp_servers, type).
        llm: Provider caller used for generation.
        current_filename: Directory holding the YAML config, forwarded to `BasePrompt`.
        substitute_llm: Optional fallback caller used when the primary `llm` call fails.
    """

    prompt_name: str = "responsibility"  # default .md file stem rendered by `run`

    def __init__(
        self,
        name: str,
        agent_config: AgentConfigs,
        llm: BaseLLM,
        current_filename: str,
        substitute_llm: BaseLLM | None = None,
    ) -> None:
        self.name = name
        self.agent_config = agent_config
        self.llm = llm
        self.current_filename = current_filename
        self.substitute_llm = substitute_llm
        self.type = agent_config.type
        self.tools = agent_config.tools
        self.mcp_servers = agent_config.mcp_servers
        self.toolbox = (
            ToolBox(current_filename=current_filename, agent_config=agent_config)
            if agent_config.tools
            else None
        )
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Set up type-specific state. Subclasses override when they need more than the defaults."""

    def build_prompt(
        self,
        question: str,
        context: str,
        agent_outputs: dict[str, str],
        prompt_name: str | None = None,
    ) -> str:
        """Render this agent's `.md` prompt with `{arguments}` filled in."""
        prompt = BasePrompt(
            current_filename=self.current_filename,
            agent_config=self.agent_config,
            question=question,
            context=context,
            agent_outputs=agent_outputs,
        )
        return prompt.get_prompt(prompt_name or self.prompt_name)

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the primary LLM, falling back to `substitute_llm` if the call raises."""
        try:
            return self.llm._call(prompt, **kwargs)
        except Exception as error:
            if not self.substitute_llm:
                raise
            logger.warning(f"Agent {self.name} falling back to substitute LLM after error: {error}")
            return self.substitute_llm._call(prompt, **kwargs)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Run one of this agent's tools by name — the other half of a provider's tool-use round trip."""
        if not self.toolbox:
            raise ValueError(f"Agent {self.name} has no tools configured.")

        return self.toolbox.call(name, arguments)

    @abstractmethod
    def run(self, question: str, context: str, agent_outputs: dict[str, str], **kwargs: Any) -> str:
        """Render the prompt, call the LLM, and return this agent's output."""


class JudgerAgent(BaseAgent):
    """Evaluates another agent's output and returns a verdict.

    Reads `judger.md` from the agent's prompt directory. That prompt references the agent being judged
    by name (e.g. `{rag_problem_thinker_agent}`), so the output under review is pulled from
    `agent_outputs` at render time.
    """

    prompt_name: str = "judger"

    def run(self, question: str, context: str, agent_outputs: dict[str, str], **kwargs: Any) -> str:
        prompt = self.build_prompt(question, context, agent_outputs)
        verdict = self._generate(prompt, **kwargs)
        logger.info(f"Judger {self.name} verdict: {verdict}")
        return verdict


class WorkerAgent(BaseAgent):
    """Generator agent that does the actual task work, optionally calling tools.

    Reads `worker.md`. This is the type behind the `generator` agents in the pipeline (model_developer,
    data_engineer, feature_preprocessing, ...), so its configured `tools` and `mcp_servers` are passed
    through to the LLM call for tool use.
    """

    prompt_name: str = "worker"

    def run(self, question: str, context: str, agent_outputs: dict[str, str], **kwargs: Any) -> str:
        prompt = self.build_prompt(question, context, agent_outputs)
        if self.toolbox:
            kwargs.setdefault("tools", self.toolbox.schemas_for(self.llm))
        if self.mcp_servers:
            kwargs.setdefault("mcp_servers", self.mcp_servers)

        output = self._generate(prompt, **kwargs)
        logger.info(f"Worker {self.name} produced output of {len(output)} chars.")
        return output


class RAGBuilderAgent(BaseAgent):
    """Retrieves supporting documents from a knowledge db, then generates over what it found.

    Reads `rag_builder.md`. `agent_config.db_vector` names the db to retrieve from; retrieval itself is
    delegated to the `db_connector` passed in at construction. Without a connector the agent falls back
    to the caller-supplied `context`, so it still runs before the db layer is wired up.
    """

    prompt_name: str = "rag_builder"

    def __init__(self, *args: Any, db_vector_connector: Any = None, db_text_connector: Any = None, embeddings_connector: Any = None, **kwargs: Any) -> None:
        self.db_vector_connector: BaseVectorConnector = db_vector_connector
        self.db_text_connector: BaseTextConnector = db_text_connector
        self.embeddings_connector: BaseEmbeddings = embeddings_connector  # Placeholder for future use if needed
        super().__init__(*args, **kwargs)

    def assembling_prompt(self, question: str, retrieved: str, context: str | None = None) -> str:
        prompt = f"""
        You are a assistant that retrieves relevant documents from a knowledge database and generates a response based on the retrieved information.
        Use the retrieved documents to answer the question. If the retrieved documents do not contain enough information to answer the question, respond with "I don't know".
        Generate with the following context. Take retrieved documents into account, but do not hallucinate information that is not present in the retrieved documents. If the retrieved documents are empty, respond with "I don't know".

        Rules: 
           - Do not make up information or provide an answer that is not supported by the retrieved documents.
           - Do not provide any information that is not present in the retrieved documents.
           - Do not provide any information that is not relevant to the question.

    
        question: {question}
        {f"context: {context}" if context else ""}

        retrieved: {retrieved}
        """
        return prompt

    def retrieve(self, question: str, context: str, top_k: int = 5) -> str:
        """Return retrieved documents for `question`, joined into a single context block."""
        if not self.db_vector_connector:
            logger.warning(f"RAGBuilder {self.name} has no db_vector_connector; using the provided context.")
            return ''

        if not self.db_text_connector:
            logger.warning(f"RAGBuilder {self.name} has no db_text_connector; returning raw vector results.")
            return ''

        if not self.embeddings_connector:
            logger.warning(f"RAGBuilder {self.name} has no embeddings_connector; using the provided context.")
            return ''

        # 1. convert question to embedding using embeddings_connector
        question_embedding = self.embeddings_connector.embed_text(question)

        # 2. retrieve relevant document IDs from the vector database
        vector_results = self.db_vector_connector.query(question_embedding, top_k=top_k)
        document_ids = [result.id for result in vector_results]
        
        # 3. fetch the actual documents from the text database using those IDs
        text_results = self.db_text_connector.query(indexes=document_ids)
        assembling_prompt = self.assembling_prompt(question, retrieved="\n\n".join(str(result) for result in text_results), context=context)
        return assembling_prompt

    def run(self, question: str, context: str, agent_outputs: dict[str, str], **kwargs: Any) -> str:
        # RAG - 1. step: retrieve relevant documents from the knowledge database
        retrieved = self.retrieve(question, context, top_k=kwargs.pop("top_k", 5))
        retrieve_prompt = self.assembling_prompt(question, retrieved, context)

        # RAG - 2. step: generate output based on the retrieved documents
        prompt = self.build_prompt(question, retrieve_prompt, agent_outputs)  # Ensure prompt is built and cached
        output = self._generate(prompt, **kwargs)
        logger.info(f"RAGBuilder {self.name} generated over db {self.agent_config.db_vector}.")
        return output


class PlannerAgent(BaseAgent):
    """Breaks a question down into an ordered plan for other agents to execute.

    Reads `planner.md`. Runs before the worker agents in a pipeline step, and its output is what those
    agents reference by this agent's name in their own prompts.
    """

    prompt_name: str = "planner"

    def run(self, question: str, context: str, agent_outputs: dict[str, str], **kwargs: Any) -> str:
        prompt = self.build_prompt(question, context, agent_outputs)
        plan = self._generate(prompt, **kwargs)
        logger.info(f"Planner {self.name} produced a plan.")
        return plan


class ClassifierAgent(BaseAgent):
    """Assigns the question (or a prior agent's output) to one of a fixed set of labels.

    Reads `classifier.md`. `labels` constrains the response: when set, an output that doesn't match one
    of them is logged as a warning, since downstream routing depends on a known label.
    """

    prompt_name: str = "classifier"
    labels: list[str] | None = None

    def __init__(self, *args: Any, labels: list[str] | None = None, **kwargs: Any) -> None:
        self.labels = labels
        super().__init__(*args, **kwargs)

    def run(self, question: str, context: str, agent_outputs: dict[str, str], **kwargs: Any) -> str:
        prompt = self.build_prompt(question, context, agent_outputs)
        label = self._generate(prompt, **kwargs).strip()

        if self.labels and label not in self.labels:
            logger.warning(f"Classifier {self.name} returned {label!r}, not one of {self.labels}.")

        logger.info(f"Classifier {self.name} classified as: {label}")
        return label
