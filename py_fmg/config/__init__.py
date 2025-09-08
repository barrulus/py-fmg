"""
Configuration modules for map generation.
"""

from .heightmap_templates import get_template, list_templates, TEMPLATES

# Avoid hard-failing on environments where pydantic-settings is not installed
# (e.g., lightweight test runs that don't use settings). Consumers that need
# settings should import from .config directly.
try:  # pragma: no cover - trivial import guard
    from .config import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None  # type: ignore


__all__ = ['get_template', 'list_templates', 'TEMPLATES', 'settings']
