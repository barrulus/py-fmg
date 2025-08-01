"""
Configuration modules for map generation.
"""

from .heightmap_templates import get_template, list_templates, TEMPLATES
from .config import settings

__all__ = ['get_template', 'list_templates', 'TEMPLATES', "settings"]