"""
Train Neural Oracle from validated labels
"""
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add AURORA to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.actions import PreprocessingAction
import xgboost as xgb

def main():
    print("="*70)
    print("TRAINING NEURAL ORACLE")
    print("="*70)
    
    # Load validated labels
    validated_path = Path('validator/validated/validated_labels.json')
    if not validated_path.exists():
        print(f"❌ Validated labels not found!")
        print("   Run llm_validator.py first!")
        return
    
    with open(validated_path, 'r') as f:
        labels = json.load(f)
    
    print(f"\n✅ Loaded {len(labels)} validated examples")
    
    # Prepare data
    X = []
    y = []
    action_set = set()
    
    for label in labels:
        features = label['features']
        action = label['action']
        
        # Convert features dict to array
        feature_array = [
            features.get('null_percentage', 0),
            features.get('unique_ratio', 0),
            features.get('skewness', 0),
            features.get('outlier_percentage', 0),
            features.get('entropy', 0),
            features.get('pattern_complexity', 0),
            features.get('multimodality_score', 0),
            features.get('cardinality_bucket', 0),
            features.get('detected_dtype', 0),
            features.get('column_name_signal', 0)
        ]
        
        X.append(feature_array)
        y.append(action)
        action_set.add(action)
    
    X = np.array(X)
    actions_list = sorted(list(action_set))
    
    # Create action encoder/decoder
    action_encoder = {i: action for i, action in enumerate(actions_list)}
    action_decoder = {action: i for i, action in enumerate(actions_list)}
    
    # Encode labels
    y_encoded = np.array([action_decoder[action] for action in y])
    
    print(f"\n📊 Dataset:")
    print(f"  Features: {X.shape}")
    print(f"  Actions: {len(actions_list)}")
    for i, action in enumerate(actions_list):
        count = sum(1 for a in y if a == action)
        print(f"    {i}: {action:30s} ({count} examples)")
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Split:")
    print(f"  Train: {X_train.shape[0]} examples")
    print(f"  Val: {X_val.shape[0]} examples")
    
    # Train XGBoost
    print(f"\n🚀 Training XGBoost...")
    
    feature_names = [
        'null_percentage', 'unique_ratio', 'skewness', 'outlier_percentage',
        'entropy', 'pattern_complexity', 'multimodality_score',
        'cardinality_bucket', 'detected_dtype', 'column_name_signal'
    ]
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': len(actions_list),
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=50,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=10
    )
    
    # Evaluate
    train_preds = model.predict(dtrain)
    train_pred_labels = np.argmax(train_preds, axis=1)
    train_accuracy = np.mean(train_pred_labels == y_train)
    
    val_preds = model.predict(dval)
    val_pred_labels = np.argmax(val_preds, axis=1)
    val_accuracy = np.mean(val_pred_labels == y_val)
    
    print(f"\n📈 Results:")
    print(f"  Train Accuracy: {train_accuracy:.1%}")
    print(f"  Val Accuracy: {val_accuracy:.1%}")
    
    # Save model
    print(f"\n💾 Saving model...")
    
    models_dir = Path('validator/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = models_dir / f'neural_oracle_v2_{timestamp}.pkl'
    
    model_data = {
        'model': model,
        'action_encoder': action_encoder,
        'action_decoder': action_decoder,
        'feature_names': feature_names,
        'metadata': {
            'version': 'v2',
            'created': timestamp,
            'num_examples': len(labels),
            'num_actions': len(actions_list),
            'train_accuracy': float(train_accuracy),
            'val_accuracy': float(val_accuracy),
            'num_boosters': 50
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved: {model_path}")
    
    # Save training history
    history_path = models_dir / 'training_history.json'
    history = {
        'date': timestamp,
        'model_path': str(model_path),
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'num_examples': len(labels),
        'num_actions': len(actions_list)
    }
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training history saved")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Train Accuracy: {train_accuracy:.1%}")
    print(f"Val Accuracy: {val_accuracy:.1%}")
    print(f"\n✅ Ready for testing!")

if __name__ == "__main__":
    main()
