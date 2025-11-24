"""
Convert AURORA theme from Dark to Light.
Modifies:
1. frontend/src/styles/design-system.css
2. frontend/src/styles/globals.css
"""

import re

def convert_design_system():
    path = 'frontend/src/styles/design-system.css'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements for design-system.css
    replacements = [
        # Backgrounds: Dark -> Light
        (r'--bg-primary: #0f172a;', r'--bg-primary: #f8fafc;'), # Slate 900 -> Slate 50
        (r'--bg-secondary: #1e293b;', r'--bg-secondary: #ffffff;'), # Slate 800 -> White
        (r'--bg-tertiary: #334155;', r'--bg-tertiary: #f1f5f9;'), # Slate 700 -> Slate 100
        (r'--bg-hover: #475569;', r'--bg-hover: #e2e8f0;'), # Slate 600 -> Slate 200

        # Text: Light -> Dark
        (r'--text-primary: #f1f5f9;', r'--text-primary: #0f172a;'), # Slate 50 -> Slate 900
        (r'--text-secondary: #cbd5e1;', r'--text-secondary: #334155;'), # Slate 300 -> Slate 700
        (r'--text-muted: #64748b;', r'--text-muted: #64748b;'), # Slate 500 (Keep same)

        # Glass Effect: Dark -> Light
        (r'--glass-bg: rgba\(30, 41, 59, 0.7\);', r'--glass-bg: rgba(255, 255, 255, 0.8);'),
        (r'--glass-border: rgba\(148, 163, 184, 0.1\);', r'--glass-border: rgba(226, 232, 240, 0.8);'),
    ]

    for old, new in replacements:
        content = re.sub(old, new, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {path}")

def convert_globals():
    path = 'frontend/src/styles/globals.css'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements for globals.css
    replacements = [
        # Body Background: Dark Gradient -> Light Gradient
        (r'background: linear-gradient\(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%\);', 
         r'background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);'),
        
        # Body Text: Slate 100 -> Slate 900
        (r'@apply bg-slate-900 text-slate-100;', r'@apply bg-slate-50 text-slate-900;'),

        # Glass Card: Dark -> Light
        (r'@apply bg-slate-800/70 backdrop-blur-xl rounded-2xl border border-slate-700/50 shadow-xl;', 
         r'@apply bg-white/80 backdrop-blur-xl rounded-2xl border border-slate-200/60 shadow-xl;'),
        
        # Glass Card Hover Border
        (r'@apply shadow-2xl border-slate-600/50;', r'@apply shadow-2xl border-blue-200/50;'),

        # Secondary Button
        (r'@apply px-6 py-3 bg-slate-700/80 backdrop-blur-sm text-slate-100 rounded-xl font-medium border border-slate-600 hover:border-slate-500', 
         r'@apply px-6 py-3 bg-white/80 backdrop-blur-sm text-slate-700 rounded-xl font-medium border border-slate-200 hover:border-slate-300'),
    ]

    for old, new in replacements:
        content = re.sub(old, new, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {path}")

if __name__ == "__main__":
    convert_design_system()
    convert_globals()
    print("Theme conversion complete!")
