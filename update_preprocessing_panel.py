"""
Script to:
1. Remove V3 Architecture panel
2. Add ExplanationModal import and state
3. Connect Explain button to open modal
"""

# Read the file
with open('frontend/src/components/PreprocessingPanel.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove V3 Architecture panel (lines 657-669)
v3_panel = '''                \u003cdiv className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border-2 border-blue-200"\u003e
                  \u003cdiv className="flex items-center gap-2 mb-2"\u003e
                    \u003cdiv className="w-2 h-2 rounded-full bg-green-500 animate-pulse"\u003e\u003c/div\u003e
                    \u003cp className="text-xs font-bold text-slate-800 uppercase"\u003eV3 Architecture\u003c/p\u003e
                  \u003c/div\u003e
                  \u003cp className="text-xs text-slate-700"\u003e
                    \u003cstrong\u003eKey Innovation:\u003c/strong\u003e Learner creates symbolic rules (not decisions).
                    After 10 corrections, new rules are injected into L1 Symbolic Engine.
                  \u003c/p\u003e
                  \u003cp className="text-xs text-blue-700 font-medium mt-2"\u003e
                    ✓ 95-99% autonomous coverage • ✓ Zero overgeneralization • ✓ All decisions traceable
                  \u003c/p\u003e
                \u003c/div\u003e'''

if v3_panel in content:
    content = content.replace(v3_panel, '')
    print("[OK] Removed V3 Architecture panel")
else:
    print("[WARNING] V3 Architecture panel not found with exact match")

# 2. Add ExplanationModal import
old_import = "import ResultCard from './ResultCard';"
new_import = """import ResultCard from './ResultCard';
import ExplanationModal from './ExplanationModal';"""

if old_import in content and 'ExplanationModal' not in content:
    content = content.replace(old_import, new_import)
    print("[OK] Added ExplanationModal import")
elif 'ExplanationModal' in content:
    print("[INFO] ExplanationModal already imported")
else:
    print("[WARNING] Could not add ExplanationModal import")

# 3. Add state for modal (after other useState declarations)
# Find the first useState and add our state after it
if "const [showCorrectionFor, setShowCorrectionFor] = useState" in content:
    old_state = "const [showCorrectionFor, setShowCorrectionFor] = useState<string | null>(null);"
    new_state = """const [showCorrectionFor, setShowCorrectionFor] = useState<string | null>(null);
  const [showExplanationFor, setShowExplanationFor] = useState<string | null>(null);
  const [explanationColumnData, setExplanationColumnData] = useState<any>(null);"""
    
    if old_state in content and 'showExplanationFor' not in content:
        content = content.replace(old_state, new_state)
        print("[OK] Added explanation modal state")
    elif 'showExplanationFor' in content:
        print("[INFO] Explanation modal state already exists")
    else:
        print("[WARNING] Could not add explanation modal state")

# 4. Replace toast with modal open
old_explain_click = "onClick={() => toast.success('Enhanced explanations coming soon!')}"
new_explain_click = """onClick={() => {
                          setShowExplanationFor(columnName);
                          setExplanationColumnData(batchResults.results[columnName]);
                        }}"""

if old_explain_click in content:
    content = content.replace(old_explain_click, new_explain_click)
    print("[OK] Connected Explain button to modal")
else:
    print("[WARNING] Could not find Explain button onClick handler")

# Write back
with open('frontend/src/components/PreprocessingPanel.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] File updated successfully!")
