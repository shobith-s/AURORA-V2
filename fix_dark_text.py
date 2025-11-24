"""
Fix hardcoded dark mode text classes in components.
Replaces text-slate-100 with text-slate-900 in specific files.
"""

import re

files_to_fix = [
    'frontend/src/components/MetricsDashboard.tsx',
    'frontend/src/components/ExplanationModal.tsx',
    'frontend/src/components/ScriptIDE.tsx',
    'frontend/src/components/Header.tsx' 
]

def fix_files():
    for path in files_to_fix:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace text-slate-100 with text-slate-900
            new_content = content.replace('text-slate-100', 'text-slate-900')
            
            # Also replace text-white with text-slate-900 ONLY in specific contexts if needed
            # For now, let's stick to slate-100 as it's the most common for body text in these files
            
            if content != new_content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Fixed {path}")
            else:
                print(f"No changes needed for {path}")
                
        except FileNotFoundError:
            print(f"File not found: {path}")

if __name__ == "__main__":
    fix_files()
