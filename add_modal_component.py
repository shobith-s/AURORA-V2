"""
Script to add ExplanationModal component to PreprocessingPanel
"""

# Read the file
with open('frontend/src/components/PreprocessingPanel.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the closing of the component (before the last closing brace and parenthesis)
old_ending = """      </div>
    </div>
  );
}"""

new_ending = """      </div>

      {/* Explanation Modal */}
      {showExplanationFor && explanationColumnData && (
        <ExplanationModal
          isOpen={!!showExplanationFor}
          onClose={() => {
            setShowExplanationFor(null);
            setExplanationColumnData(null);
          }}
          columnData={explanationColumnData}
        />
      )}
    </div>
  );
}"""

if old_ending in content:
    content = content.replace(old_ending, new_ending)
    print("[OK] Added ExplanationModal component")
else:
    print("[WARNING] Could not find component ending")
    # Try to find if modal is already added
    if 'ExplanationModal' in content and 'showExplanationFor' in content:
        print("[INFO] ExplanationModal component may already be added")

# Write back
with open('frontend/src/components/PreprocessingPanel.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] File updated successfully!")
