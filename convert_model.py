"""
Convert v2 ensemble model to NeuralOracle format
Simple wrapper - no over-engineering
"""
import pickle
from pathlib import Path
from src.core.actions import PreprocessingAction

# Load the v2 ensemble model
v2_model_path = Path('models/neural_oracle_v2_improved_20251129_150244.pkl')

print(f"Loading v2 model from: {v2_model_path}")
with open(v2_model_path, 'rb') as f:
    ensemble_model = pickle.load(f)

print(f"Model type: {type(ensemble_model)}")
print(f"Model loaded successfully")

# Create action encoder/decoder (same as training)
all_actions = [
    'keep_as_is', 'drop', 'fill_mean', 'fill_median', 'fill_mode',
    'fill_forward', 'fill_backward', 'fill_zero', 'standard_scale',
    'minmax_scale', 'robust_scale', 'log_transform', 'sqrt_transform',
    'box_cox', 'yeo_johnson', 'quantile_transform', 'power_transform',
    'onehot_encode', 'label_encode', 'ordinal_encode', 'target_encode',
    'frequency_encode', 'binary_encode', 'hash_encode'
]

action_encoder = {i: action for i, action in enumerate(all_actions)}
action_decoder = {action: i for i, action in enumerate(all_actions)}

# Feature names (20 features from v2)
feature_names = [
    'null_percentage', 'unique_ratio', 'skewness', 'outlier_percentage',
    'entropy', 'pattern_complexity', 'multimodality_score',
    'cardinality_bucket', 'detected_dtype', 'column_name_signal',
    'kurtosis', 'coefficient_of_variation', 'zero_ratio',
    'has_negative', 'has_decimal', 'name_contains_id',
    'name_contains_date', 'name_contains_price', 'range_ratio', 'iqr_ratio'
]

# Wrap in NeuralOracle format
save_dict = {
    'model': ensemble_model,
    'action_encoder': action_encoder,
    'action_decoder': action_decoder,
    'feature_names': feature_names
}

# Save as neural_oracle_v1.pkl (what the system looks for)
output_path = Path('models/neural_oracle_v1.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(save_dict, f)

print(f"\n✅ Converted model saved to: {output_path}")
print(f"   Size: {output_path.stat().st_size / 1024:.0f} KB")
print(f"\nNow restart the backend to use the 89.4% model!")
