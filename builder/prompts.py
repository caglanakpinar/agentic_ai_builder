import re
from pathlib import Path

from uilts.configs import Configs, AgentConfigs
from uilts.logger import logger


PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")  # matches {argument_name} placeholders in a .md prompt


class BasePrompt(Configs):
    """Reads an agent's `.md` prompt files and renders them by filling in their `{arguments}`.

    An agent says where its prompts live in one of two ways:
      - `AgentConfigs.prompt` is a directory containing one `.md` file per prompt (e.g. `worker.md`,
        `judger.md`), loaded into a `prompt_name -> markdown` mapping keyed by file stem;
      - `AgentConfigs.prompt_path` is a single `.md` file, for an agent that only ever renders one. It is
        registered under its own stem and also serves whichever prompt name is asked for, so an agent
        looking for `classifier` finds it without the file having to be named that.

    Either path is resolved as given — absolute, or relative to where the command was run — falling back
    to being read relative to the config directory. `read_prompts` also collects every `{placeholder}`
    each file declares.

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
        self.prompt_dir = self.resolve_path(agent_config.prompt)
        self.prompt_file = self.resolve_path(agent_config.prompt_path)
        self.single_prompt: str | None = None  # set when the agent configured one file rather than a directory
        self.arguments = agent_config.arguments
        self.question = question
        self.context = context
        self.agent_outputs = agent_outputs
        self.threshold_values: dict[str, object] = dict(agent_config.thresholds or {})
        self.thresholds = self.render_thresholds()  # what `{thresholds}` renders as
        self.dependencies = (
            [agent_config.dependency_agent]
            if isinstance(agent_config.dependency_agent, str)
            else list(agent_config.dependency_agent or [])
        )
        self.agent_output = self.render_agent_output()  # what `{agent_output}` renders as
        self.read_prompts()

    def render_agent_output(self) -> str:
        """Render the work this agent depends on, as the block `{agent_output}` stands for.

        A prompt that names each upstream agent itself (`{data_engineer}`, `{evaluator}`, ...) has the
        pipeline's wiring written into it twice — once in the config and once in the markdown — so moving
        a step means editing both. Declaring `dependency_agent` in the config instead lets the prompt ask
        for "what the agents before you produced" and get exactly that, labelled by who produced it.

        An upstream agent that has not run yet is said so explicitly rather than left blank: a prompt
        that silently loses a section reads as though there was nothing to say.
        """
        if not self.dependencies:
            return "(nothing upstream — this agent starts the work)"

        blocks = []
        for name in self.dependencies:
            output = self.agent_outputs.get(name)
            if output is None:
                logger.warning(f"Dependency {name} has not produced an output yet.")
                output = f"(no output — {name} has not run yet)"
            blocks.append(f"### {name}\n\n{output}")

        return "\n\n".join(blocks)

    def render_thresholds(self) -> str:
        """Render the agent's `thresholds` as the block a judger's prompt states its bars in.

        A `min_`/`max_` prefix carries the direction, so the rendered line reads as the rule it is —
        the judge is told the bar, not just the number, and each one is also available on its own as
        `{min_holdout_f1}` and the like.
        """
        if not self.threshold_values:
            return "(none configured)"

        lines = []
        for name, value in self.threshold_values.items():
            direction = "at least" if name.startswith("min_") else "at most" if name.startswith("max_") else "exactly"
            measure = name.split("_", 1)[1] if name.startswith(("min_", "max_")) else name
            lines.append(f"  - {measure.replace('_', ' ')}: {direction} {value}   (`{name}`)")

        return "\n".join(lines)

    def resolve_path(self, path: str | None) -> Path | None:
        """Resolve a configured prompt path: as given when it exists, otherwise under the config directory."""
        if not path:
            return None

        given = Path(path)
        return given if given.exists() else self.current_dir / path

    def read_prompts(self) -> None:
        """Load the agent's prompt file, or every `.md` in its prompt directory, with their arguments."""
        files: list[Path] = []
        if self.prompt_file and self.prompt_file.is_file():
            files = [self.prompt_file]
        elif self.prompt_dir and self.prompt_dir.is_dir():
            files = sorted(path for path in self.prompt_dir.glob("*.md") if path.is_file())
        else:
            logger.warning(f"No prompt found at {self.prompt_file or self.prompt_dir}.")
            return

        for prompt_file in files:
            markdown = prompt_file.read_text().strip()
            self.prompt_configs[prompt_file.stem] = markdown
            self.prompt_arguments[prompt_file.stem] = list(
                dict.fromkeys(PLACEHOLDER_PATTERN.findall(markdown))
            )

        if self.prompt_file:  # one configured file answers to whatever prompt name the agent asks for
            self.single_prompt = self.prompt_configs[self.prompt_file.stem]

        discovered = {arg for args in self.prompt_arguments.values() for arg in args}
        self.arguments = sorted(set(self.arguments) | discovered)

    def get_raw_prompt(self, prompt_name: str) -> str:
        """Return a prompt's markdown with its `{arguments}` left unfilled."""
        return self.prompt_configs.get(prompt_name) or self.single_prompt or ''

    def get_arguments(self, prompt_name: str) -> list[str]:
        """Return the argument names a single prompt declares."""
        return self.prompt_arguments.get(prompt_name, [])

    def get_prompt(self, prompt_name: str) -> str:
        """Return a prompt's markdown with its `{arguments}` filled in."""
        markdown = self.get_raw_prompt(prompt_name)
        if not markdown:
            logger.warning(f"No prompt named {prompt_name} found in {self.prompt_file or self.prompt_dir}.")
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

        if argument in self.threshold_values:  # a single bar referenced by name, e.g. {min_holdout_f1}
            return str(self.threshold_values[argument])

        logger.warning(f"No value provided for argument {argument}.")
        return match.group(0)
