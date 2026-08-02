from abc import abstractmethod
from typing import Any

from builder.prompts import BasePrompt
from models.llms import BaseLLM
from uilts.configs import AgentConfigs
from uilts.logger import logger


class BaseAgent:
    """Common interface for the agent types a pipeline step can run (judger, thinker, classifier, generator, ...).

    An agent pairs an `AgentConfigs` entry with a live `BaseLLM` caller. Subclasses implement:
      - `_initialize_agent()`: set up any type-specific state (defaults to a no-op).
      - `run(question, context, agent_outputs)`: render this agent's prompt, call the LLM, return the
        agent's output as plain text.

    Prompt rendering is delegated to `BasePrompt`, so an agent's `.md` prompt can reference `{question}`,
    `{context}`, or another agent by name — the latter resolved from `agent_outputs`, which is how one
    step's result feeds the next.

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
        if self.tools:
            kwargs.setdefault("tools", self.tools)
        if self.mcp_servers:
            kwargs.setdefault("mcp_servers", self.mcp_servers)

        output = self._generate(prompt, **kwargs)
        logger.info(f"Worker {self.name} produced output of {len(output)} chars.")
        return output


class RAGBuilderAgent(BaseAgent):
    """Retrieves supporting documents from a knowledge db, then generates over what it found.

    Reads `rag_builder.md`. `agent_config.db` names the db to retrieve from; retrieval itself is
    delegated to the `db_connector` passed in at construction. Without a connector the agent falls back
    to the caller-supplied `context`, so it still runs before the db layer is wired up.
    """

    prompt_name: str = "rag_builder"

    def __init__(self, *args: Any, db_connector: Any = None, **kwargs: Any) -> None:
        self.db_connector = db_connector
        super().__init__(*args, **kwargs)

    def retrieve(self, question: str, top_k: int = 5) -> str:
        """Return retrieved documents for `question`, joined into a single context block."""
        if not self.db_connector:
            logger.warning(f"RAGBuilder {self.name} has no db_connector; using the provided context.")
            return ''

        results = self.db_connector.query(question, top_k=top_k)
        return "\n\n".join(str(result) for result in results)

    def run(self, question: str, context: str, agent_outputs: dict[str, str], **kwargs: Any) -> str:
        retrieved = self.retrieve(question, top_k=kwargs.pop("top_k", 5))
        prompt = self.build_prompt(question, retrieved or context, agent_outputs)
        output = self._generate(prompt, **kwargs)
        logger.info(f"RAGBuilder {self.name} generated over db {self.agent_config.db}.")
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
