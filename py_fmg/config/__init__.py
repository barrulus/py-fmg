"""
Configuration modules for map generation.
"""

from .heightmap_templates import get_template, list_templates, TEMPLATES

# Import settings from parent level
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from config import settings
except ImportError:
    # Fallback if config.py doesn't exist
    settings = None

__all__ = ['get_template', 'list_templates', 'TEMPLATES', 'settings']