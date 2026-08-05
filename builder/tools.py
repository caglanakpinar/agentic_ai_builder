import importlib
import inspect
from dataclasses import asdict
from typing import Any, Callable

from utils.configs import AgentConfigs, Configs, ToolConfigs
from utils.logger import logger


# All three come from `utils/default_config.yaml`: the type names a tool's `args[].type` accepts, the
# providers whose tool schemas differ from OpenAI's, and the dialect everyone else speaks.
JSON_SCHEMA_TYPES: dict[str, str] = Configs.registry("json_schema_types")
PROVIDER_BY_LLM: dict[str, str] = Configs.registry("tool_dialects")
DEFAULT_PROVIDER: str = Configs.setting("tool_dialect")


def import_function(caller: str, tool_name: str) -> Callable[..., Any] | None:
    """Import the Python function a tool's `caller` path points at.

    `caller` is accepted in either form:
      - a module path, where the function is named after the tool: `"benchmarks.functions"` imports
        `benchmarks.functions.<tool_name>`,
      - the full dotted path to the function itself: `"benchmarks.functions.data_ingestion_caller"`.

    The module form is tried first, so a module that does export a function named after the tool wins;
    otherwise the path is split at its last dot and the tail is looked up as an attribute.

    Returns None (and logs why) when the module is missing, the attribute is absent, or it isn't
    callable, so one broken path doesn't take the rest of an agent's tools down with it.
    """
    candidates: list[tuple[str, str]] = [(caller, tool_name)]
    if "." in caller:
        module_path, attribute = caller.rsplit(".", 1)
        candidates.append((module_path, attribute))

    failure = f"no module named {caller!r}"
    for module_path, attribute in candidates:
        try:
            module = importlib.import_module(module_path)
        except ImportError as error:
            failure = f"cannot import module {module_path!r} ({error})"
            continue

        function = getattr(module, attribute, None)
        if callable(function):
            logger.info(f"Tool {tool_name} bound to {module_path}.{attribute}.")
            return function

        failure = (
            f"module {module_path!r} has no callable {attribute!r}"
            if function is None
            else f"{module_path}.{attribute} is not callable"
        )

    logger.error(f"Tool {tool_name}: {failure} for caller path {caller!r}.")
    return None


class Tool(ToolConfigs):
    """One configured tool: the Python function behind it, plus the schema its agent's LLM expects.

    `caller` says where the function lives and is resolved by `import_function`, either as a module
    path whose function is named after the tool (`"benchmarks.functions"` -> `<module>.<name>`) or as
    the full dotted path to the function itself. A tool with no `caller` falls back to
    `default_module`, which is how a whole `tools:` block can be pointed at a single module of
    functions keyed by tool name.

    `args` — each entry a `{name, type, description}` mapping from the YAML — becomes the JSON Schema
    the provider validates tool calls against. An argument is required unless it declares
    `required: false` or carries a `default`. Optional `enum` and `items` keys are passed through.

    `schema(provider)` renders that in the dialect one provider expects ("anthropic", "google", or the
    OpenAI-compatible default), and `run(**arguments)` executes the imported function after checking
    the arguments against the same declaration.

    Args:
        name: Tool name as it appears in the YAML `tools:` block, and the name the LLM calls it by.
        description: What the tool does; this is what the model reads when deciding to call it.
        type: Role this tool plays ("generator", "retriever", "tool caller", "tool generator").
        llm: Name of the LLM backing an LLM-implemented tool, if any.
        embeddings: Name of the embeddings config this tool uses, if any.
        config: Free-form extra settings for the tool.
        caller: Import path of the function to call, in either form described above.
        args: Declared arguments, each a `{name, type, description}` mapping.
        default_module: Module to import from when `caller` is not set.
    """

    function: Callable[..., Any] | None = None  # the imported callable, None when it can't be resolved

    def __init__(
        self,
        name: str,
        description: str = '',
        type: str = "tool caller",
        llm: str | None = None,
        embeddings: str | None = None,
        config: dict[str, str] | None = None,
        caller: str | None = None,
        args: list[dict[str, str]] | None = None,
        default_module: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            type=type,
            llm=llm,
            embeddings=embeddings,
            config=config,
            caller=caller,
            args=args,
        )
        self.default_module = default_module
        self.function = self.resolve_function()

    def resolve_function(self) -> Callable[..., Any] | None:
        """Import the function behind this tool, from `caller` or else from `default_module`."""
        caller = self.caller or self.default_module
        if not caller:
            logger.warning(
                f"Tool {self.name} declares no caller and no default module; "
                f"it has no function to call{' (llm-backed tool)' if self.llm else ''}."
            )
            return None

        return import_function(caller, self.name)

    def parameters(self) -> dict[str, Any]:
        """Render `args` as the JSON Schema object every provider wants its arguments described in."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for arg in self.args or []:
            argument = arg.get("name")
            if not argument:
                logger.warning(f"Tool {self.name} declares an argument without a name; skipping it.")
                continue

            declared = str(arg.get("type", "str")).lower()
            if declared not in JSON_SCHEMA_TYPES:
                logger.warning(f"Tool {self.name} argument {argument} has unknown type {declared!r}; using string.")

            schema: dict[str, Any] = {
                "type": JSON_SCHEMA_TYPES.get(declared, "string"),
                "description": arg.get("description", ''),
            }
            if arg.get("enum"):
                schema["enum"] = arg["enum"]
            if schema["type"] == "array":
                schema["items"] = arg.get("items", {"type": "string"})
            if "default" in arg:
                schema["default"] = arg["default"]

            properties[argument] = schema
            if arg.get("required", "default" not in arg):
                required.append(argument)

        return {"type": "object", "properties": properties, "required": required}

    def declaration(self) -> dict[str, Any]:
        """Return the `{name, description, parameters}` triple every dialect is built out of."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters()}

    def schema(self, provider: str = DEFAULT_PROVIDER) -> dict[str, Any]:
        """Return this tool declared in `provider`'s dialect, ready to pass as an LLM `tools` entry."""
        if provider == "anthropic":
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self.parameters(),
            }

        if provider == "google":
            return {"function_declarations": [self.declaration()]}

        return {"type": "function", "function": self.declaration()}

    def run(self, **arguments: Any) -> Any:
        """Call the imported function, after checking `arguments` against the declared `args`."""
        if not self.function:
            raise ValueError(
                f"Tool {self.name} has no function to call; check its caller path {self.caller!r}."
            )

        arguments = self.checked_arguments(arguments)
        logger.info(f"Calling tool {self.name} with arguments {sorted(arguments)}.")
        return self.function(**arguments)

    __call__ = run

    def checked_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Fail on missing required arguments, and drop ones the function cannot accept."""
        missing = [
            argument for argument in self.parameters()["required"] if argument not in arguments
        ]
        if missing:
            raise ValueError(f"Tool {self.name} is missing required argument(s): {', '.join(missing)}.")

        try:
            parameters = inspect.signature(self.function).parameters
        except (TypeError, ValueError):  # builtins and C functions expose no signature
            return arguments

        if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()):
            return arguments

        accepted = {name: value for name, value in arguments.items() if name in parameters}
        for name in arguments.keys() - accepted.keys():
            logger.warning(f"Tool {self.name} was passed {name!r}, which its function does not accept; dropping it.")
        return accepted


class ToolBox(Configs):
    """Builds the tools one agent declares, ready to hand to its LLM.

    Each entry of `AgentConfigs.tools` is either the name of a tool defined in the YAML `tools:` block,
    or an inline mapping that defines one (and may override fields of a configured tool of the same
    name). Every entry is resolved to a `Tool`, whose function is imported at build time — a tool whose
    function cannot be imported is logged and left out, so the model is only ever offered tools that
    can actually run.

    `schemas(provider)` (or `schemas_for(llm)`, which picks the dialect from the caller's class) returns
    the list to pass as the LLM's `tools`, and `call(name, arguments)` executes what the model asked
    for, which is the other half of a tool-use round trip.

    Args:
        current_filename: Directory holding the YAML config, forwarded to `Configs`.
        agent_config: The agent whose `tools` entries this instance builds.
        default_module: Module to import from for tools that declare no `caller`, e.g.
            `"benchmarks.functions"` to resolve every tool against a function named after it.
    """

    def __init__(
        self,
        current_filename: str,
        agent_config: AgentConfigs,
        default_module: str | None = None,
    ) -> None:
        super().__init__(current_filename)
        self.agent_config = agent_config
        self.default_module = default_module
        self.agent_tools: dict[str, Tool] = {}  # tool_name -> Tool, in the order the agent declares them
        self.build_tools()

    def build_tools(self) -> None:
        """Resolve every entry of `agent_config.tools` into an importable, callable `Tool`."""
        for entry in self.agent_config.tools or []:
            tool_config = self.resolve_config(entry)
            if not tool_config:
                continue

            tool = Tool(**asdict(tool_config), default_module=self.default_module)
            if not tool.function:
                logger.warning(f"Tool {tool.name} has no importable function; leaving it out of the agent's tools.")
                continue

            self.agent_tools[tool.name] = tool

    def resolve_config(self, entry: str | dict[str, Any]) -> ToolConfigs | None:
        """Resolve one `tools` entry: a name from the YAML `tools:` block, or an inline definition."""
        if isinstance(entry, str):
            entry = {"name": entry}

        name = entry.get("name", '')
        configured = self.tool_configs.get(name)
        if not configured and len(entry) == 1:
            logger.warning(f"Tool {name!r} is not defined in the `tools:` config; skipping it.")
            return None

        fields = (
            asdict(configured)
            if configured
            else {"name": name, "description": '', "type": "tool caller"}
        )
        for key, value in entry.items():
            if key not in ToolConfigs.__dataclass_fields__:
                logger.warning(f"Tool {name} declares unknown field {key!r}; ignoring it.")
                continue
            fields[key] = value

        return ToolConfigs(**fields)

    def schemas(self, provider: str = DEFAULT_PROVIDER) -> list[dict[str, Any]]:
        """Return every built tool declared in `provider`'s dialect, as the LLM's `tools` argument."""
        if provider == "google":  # Gemini takes one tool holding all of the function declarations
            declarations = [tool.declaration() for tool in self.agent_tools.values()]
            return [{"function_declarations": declarations}] if declarations else []

        return [tool.schema(provider) for tool in self.agent_tools.values()]

    def schemas_for(self, llm: Any) -> list[dict[str, Any]]:
        """Return the schemas in the dialect `llm`'s provider expects, picked from its caller class."""
        provider = PROVIDER_BY_LLM.get(type(llm).__name__, DEFAULT_PROVIDER)
        return self.schemas(provider)

    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return the imported callables keyed by tool name, for callers that run the loop themselves."""
        return {name: tool.function for name, tool in self.agent_tools.items() if tool.function}

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Run the tool the model asked for, by name, with the arguments it supplied."""
        tool = self.agent_tools.get(name)
        if not tool:
            raise KeyError(f"No tool named {name!r}; this agent has {sorted(self.agent_tools) or 'none'}.")

        return tool.run(**(arguments or {}))

    def __getitem__(self, name: str) -> Tool:
        return self.agent_tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self.agent_tools

    def __iter__(self):
        return iter(self.agent_tools.values())

    def __len__(self) -> int:
        return len(self.agent_tools)
