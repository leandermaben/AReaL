"""Agent tools for audio search.

Each tool is a module exporting a class that subclasses Tool.
Register new tools by adding them to TOOL_REGISTRY below.
"""

from .base import Tool
from .clap_search import CLAPSearchTool
from .omni_probe import OmniProbeTool
from .submit import SubmitTool
from .transcript_search import TranscriptSearchTool

# Plain dict — no magic. Add new tools here as they're implemented.
TOOL_REGISTRY: dict[str, type[Tool]] = {
    "clap_search": CLAPSearchTool,
    "omni_probe": OmniProbeTool,
    "submit": SubmitTool,
    "transcript_search": TranscriptSearchTool,
}


def get_all_tool_schemas() -> list[dict]:
    """Return OpenAI-style function schemas for all registered tools."""
    return [cls.schema() for cls in TOOL_REGISTRY.values()]
