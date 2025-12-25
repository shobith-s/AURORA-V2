"""
Pipeline exporter for generating reproducible preprocessing code.
Converts preprocessing decisions into executable code/configs.
"""

from typing import List, Dict, Any
import json
import pickle
from io import BytesIO
from ..core.actions import PreprocessingAction


class PipelineExporter:
    """Export preprocessing pipelines in multiple formats."""
    
    def __init__(self):
        """Initialize the pipeline exporter."""
        self.action_to_sklearn = {
            PreprocessingAction.STANDARD_SCALE: "StandardScaler",
            PreprocessingAction.ROBUST_SCALE: "RobustScaler",
            PreprocessingAction.MINMAX_SCALE: "MinMaxScaler",
            PreprocessingAction.LOG_TRANSFORM: "FunctionTransformer",
            PreprocessingAction.ONEHOT_ENCODE: "OneHotEncoder",
            PreprocessingAction.LABEL_ENCODE: "LabelEncoder",
            PreprocessingAction.ORDINAL_ENCODE: "OrdinalEncoder",
        }
    
    def export_python_code(self, decisions: List[Dict[str, Any]]) -> str:
        """
        Generate standalone Python function from preprocessing decisions.
        
        Args:
            decisions: List of preprocessing decisions with column names and actions
            
        Returns:
            Python code as string
        """
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler",
            "from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder",
            "",
            "",
            "def preprocess(df):",
            '    """Auto-generated preprocessing function by AURORA V2."""',
            "    df = df.copy()",
            ""
        ]
        
        # Group decisions by action type
        drops = []
        transforms = []
        
        for decision in decisions:
            column_name = decision.get('column_name', '')
            action = decision.get('action', '')
            
            if not column_name or not action:
                continue
            
            # Convert action string to enum if needed
            if isinstance(action, str):
                try:
                    action_enum = PreprocessingAction(action)
                except ValueError:
                    continue
            else:
                action_enum = action
            
            # Generate code based on action
            if action_enum == PreprocessingAction.DROP_COLUMN:
                drops.append(column_name)
            
            elif action_enum == PreprocessingAction.STANDARD_SCALE:
                transforms.append(
                    f"    df['{column_name}'] = StandardScaler().fit_transform(df[['{column_name}']])"
                )
            
            elif action_enum == PreprocessingAction.ROBUST_SCALE:
                transforms.append(
                    f"    df['{column_name}'] = RobustScaler().fit_transform(df[['{column_name}']])"
                )
            
            elif action_enum == PreprocessingAction.MINMAX_SCALE:
                transforms.append(
                    f"    df['{column_name}'] = MinMaxScaler().fit_transform(df[['{column_name}']])"
                )
            
            elif action_enum == PreprocessingAction.LOG_TRANSFORM:
                transforms.append(
                    f"    df['{column_name}'] = np.log(df['{column_name}'] + 1)  # log1p transform"
                )
            
            elif action_enum == PreprocessingAction.LABEL_ENCODE:
                transforms.append(
                    f"    df['{column_name}'] = LabelEncoder().fit_transform(df['{column_name}'])"
                )
            
            elif action_enum == PreprocessingAction.ONEHOT_ENCODE:
                transforms.append(
                    f"    # One-hot encode '{column_name}' (creates multiple columns)"
                )
                transforms.append(
                    f"    df = pd.get_dummies(df, columns=['{column_name}'], prefix='{column_name}')"
                )
            
            elif action_enum == PreprocessingAction.FILL_NULL_MEAN:
                transforms.append(
                    f"    df['{column_name}'].fillna(df['{column_name}'].mean(), inplace=True)"
                )
            
            elif action_enum == PreprocessingAction.FILL_NULL_MEDIAN:
                transforms.append(
                    f"    df['{column_name}'].fillna(df['{column_name}'].median(), inplace=True)"
                )
            
            elif action_enum == PreprocessingAction.FILL_NULL_MODE:
                transforms.append(
                    f"    df['{column_name}'].fillna(df['{column_name}'].mode()[0], inplace=True)"
                )
        
        # Add transforms
        if transforms:
            code_lines.append("    # Apply transformations")
            code_lines.extend(transforms)
            code_lines.append("")
        
        # Add drops at the end
        if drops:
            code_lines.append("    # Drop columns")
            drops_str = ", ".join([f"'{col}'" for col in drops])
            code_lines.append(f"    df = df.drop(columns=[{drops_str}])")
            code_lines.append("")
        
        code_lines.extend([
            "    return df",
            "",
            "",
            "# Usage:",
            "# df_preprocessed = preprocess(df)",
        ])
        
        return "\n".join(code_lines)
    
    def export_sklearn_pipeline(self, decisions: List[Dict[str, Any]]) -> bytes:
        """
        Generate serialized sklearn Pipeline from preprocessing decisions.
        
        Args:
            decisions: List of preprocessing decisions
            
        Returns:
            Pickled sklearn Pipeline as bytes
        """
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
            from sklearn.compose import ColumnTransformer
            
            transformers = []
            
            for decision in decisions:
                column_name = decision.get('column_name', '')
                action = decision.get('action', '')
                
                if not column_name or not action:
                    continue
                
                # Convert action string to enum
                if isinstance(action, str):
                    try:
                        action_enum = PreprocessingAction(action)
                    except ValueError:
                        continue
                else:
                    action_enum = action
                
                # Skip drop actions (handle separately)
                if action_enum == PreprocessingAction.DROP_COLUMN:
                    continue
                
                # Map to sklearn transformer
                if action_enum == PreprocessingAction.STANDARD_SCALE:
                    transformers.append((f'scale_{column_name}', StandardScaler(), [column_name]))
                
                elif action_enum == PreprocessingAction.ROBUST_SCALE:
                    transformers.append((f'robust_{column_name}', RobustScaler(), [column_name]))
                
                elif action_enum == PreprocessingAction.MINMAX_SCALE:
                    transformers.append((f'minmax_{column_name}', MinMaxScaler(), [column_name]))
            
            if not transformers:
                # Return empty pipeline if no transformers
                pipeline = Pipeline([('passthrough', 'passthrough')])
            else:
                # Create ColumnTransformer
                ct = ColumnTransformer(transformers, remainder='passthrough')
                pipeline = Pipeline([('preprocessing', ct)])
            
            # Serialize to bytes
            buffer = BytesIO()
            pickle.dump(pipeline, buffer)
            return buffer.getvalue()
        
        except ImportError:
            # If sklearn not available, return error message as bytes
            error_msg = "Error: sklearn not installed. Cannot generate pipeline."
            return error_msg.encode('utf-8')
    
    def export_json_config(self, decisions: List[Dict[str, Any]]) -> str:
        """
        Generate JSON configuration from preprocessing decisions.
        
        Args:
            decisions: List of preprocessing decisions
            
        Returns:
            JSON string
        """
        config = {
            "version": "1.0",
            "generated_by": "AURORA V2",
            "columns": {}
        }
        
        for decision in decisions:
            column_name = decision.get('column_name', '')
            action = decision.get('action', '')
            confidence = decision.get('confidence', 0.0)
            explanation = decision.get('explanation', '')
            
            if not column_name:
                continue
            
            # Convert action enum to string if needed
            if isinstance(action, PreprocessingAction):
                action_str = action.value
            else:
                action_str = str(action)
            
            config["columns"][column_name] = {
                "action": action_str,
                "confidence": round(confidence, 3),
                "explanation": explanation,
                "parameters": decision.get('parameters', {})
            }
        
        return json.dumps(config, indent=2)
    
    def export(self, decisions: List[Dict[str, Any]], format: str = "python") -> Any:
        """
        Export preprocessing pipeline in specified format.
        
        Args:
            decisions: List of preprocessing decisions
            format: Export format ('python', 'sklearn', 'json')
            
        Returns:
            Exported pipeline (str for python/json, bytes for sklearn)
        """
        if format == "python":
            return self.export_python_code(decisions)
        elif format == "sklearn":
            return self.export_sklearn_pipeline(decisions)
        elif format == "json":
            return self.export_json_config(decisions)
        else:
            raise ValueError(f"Unknown export format: {format}")
