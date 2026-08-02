import re

from uilts.configs import Configs, AgentConfigs
from uilts.logger import logger


PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")  # matches {argument_name} placeholders in a .md prompt


class BasePrompt(Configs):
    """Reads an agent's `.md` prompt files and renders them by filling in their `{arguments}`.

    `AgentConfigs.prompt` holds a path (relative to `self.current_dir`) to a directory containing one
    `.md` file per prompt (e.g. `tool_caller.md`, `tool_generator.md`). `read_prompts` loads that
    directory into a `prompt_name -> markdown` mapping and collects every `{placeholder}` each file
    declares.

    `get_prompt` renders one of those prompts. Each `{argument}` is resolved in order:
      1. an argument naming another configured agent is filled from `agent_outputs`, i.e. that agent's
         produced output,
      2. otherwise it is filled from a matching attribute on this instance (`question`, `context`).
    Anything unresolved is left in place and logged, so a partially rendered prompt stays readable.

    Args:
        current_filename: Directory holding the YAML config, forwarded to `Configs`.
        agent_config: The agent whose prompt directory and arguments this instance renders.
        question: Value for a `{question}` placeholder.
        context: Value for a `{context}` placeholder.
        agent_outputs: Outputs produced by already-run agents, keyed by agent name.
    """

    def __init__(
        self,
        current_filename: str,
        agent_config: AgentConfigs,
        question: str,
        context: str,
        agent_outputs: dict[str, str],
    ):
        super().__init__(current_filename)
        self.prompt_configs: dict[str, str] = {}  # prompt_name -> markdown text
        self.prompt_arguments: dict[str, list[str]] = {}  # prompt_name -> [argument_name, ...]
        self.prompt_dir = self.current_dir / agent_config.prompt
        self.arguments = agent_config.arguments
        self.question = question
        self.context = context
        self.agent_outputs = agent_outputs
        self.read_prompts()

    def read_prompts(self) -> None:
        """Load every `.md` file in `prompt_dir` and collect the arguments each one declares."""
        if not self.prompt_dir.is_dir():
            logger.warning(f"Prompt directory {self.prompt_dir} does not exist.")
            return

        for prompt_file in self.prompt_dir.glob("*.md"):
            if not prompt_file.is_file():
                continue

            markdown = prompt_file.read_text().strip()
            self.prompt_configs[prompt_file.stem] = markdown
            self.prompt_arguments[prompt_file.stem] = list(
                dict.fromkeys(PLACEHOLDER_PATTERN.findall(markdown))
            )

        discovered = {arg for args in self.prompt_arguments.values() for arg in args}
        self.arguments = sorted(set(self.arguments) | discovered)

    def get_raw_prompt(self, prompt_name: str) -> str:
        """Return a prompt's markdown with its `{arguments}` left unfilled."""
        return self.prompt_configs.get(prompt_name, '')

    def get_arguments(self, prompt_name: str) -> list[str]:
        """Return the argument names a single prompt declares."""
        return self.prompt_arguments.get(prompt_name, [])

    def get_prompt(self, prompt_name: str) -> str:
        """Return a prompt's markdown with its `{arguments}` filled in."""
        markdown = self.get_raw_prompt(prompt_name)
        if not markdown:
            logger.warning(f"No prompt named {prompt_name} found in {self.prompt_dir}.")
            return ''

        return PLACEHOLDER_PATTERN.sub(self._substitute, markdown)

    def _substitute(self, match: re.Match) -> str:
        """Resolve one `{argument}`: agent-named ones come from `agent_outputs`, the rest from self."""
        argument = match.group(1)

        if argument in self.agent_configs:
            if argument in self.agent_outputs:
                return str(self.agent_outputs[argument])
            logger.warning(f"Agent {argument} has not produced an output yet.")
            return match.group(0)

        value = getattr(self, argument, None)
        if value is not None:
            return str(value)

        logger.warning(f"No value provided for argument {argument}.")
        return match.group(0)
