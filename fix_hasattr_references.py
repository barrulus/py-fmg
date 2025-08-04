#!/usr/bin/env python3
"""
Script pour corriger automatiquement les références hasattr vers les nouvelles propriétés tile_events.
"""

import re
import os

def fix_hasattr_references(file_path):
    """Corriger les références hasattr dans un fichier."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Patterns à corriger
    patterns = [
        # hasattr(self.graph, "property") -> self.graph.property is not None
        (r'hasattr\(self\.graph,\s*["\'](\w+)["\']\)', r'self.graph.\1 is not None'),
        
        # hasattr(self.graph, "property") and condition -> self.graph.property is not None and condition
        (r'hasattr\(self\.graph,\s*["\'](\w+)["\']\)\s*and\s*([^)]+)\)', r'self.graph.\1 is not None and \2'),
        
        # if hasattr(self.graph, "property") and i < len(self.graph.property):
        (r'if\s+hasattr\(self\.graph,\s*["\'](\w+)["\']\)\s*and\s*(\w+)\s*<\s*len\(self\.graph\.(\w+)\):', 
         r'if self.graph.\1 is not None and \2 < len(self.graph.\3):'),
    ]
    
    changes_made = 0
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes_made += 1
            content = new_content
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Fixed {changes_made} patterns in {file_path}")
        return True
    else:
        print(f"⚪ No changes needed in {file_path}")
        return False

def main():
    """Corriger tous les fichiers core."""
    core_files = [
        'py_fmg/core/settlements.py',
        'py_fmg/core/religions.py',
        'py_fmg/core/hydrology.py',
        'py_fmg/core/biomes.py',
        'py_fmg/core/cultures.py'
    ]
    
    total_fixed = 0
    for file_path in core_files:
        if os.path.exists(file_path):
            if fix_hasattr_references(file_path):
                total_fixed += 1
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"\n🎯 Summary: Fixed {total_fixed} files")

if __name__ == "__main__":
    main()
