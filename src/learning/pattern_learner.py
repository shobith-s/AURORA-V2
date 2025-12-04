"""
Pattern learner for adaptive learning from user corrections.
Extracts patterns from corrections and creates new rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import hashlib
import json
import numpy as np
from pathlib import Path

from ..core.actions import PreprocessingAction


@dataclass
class CorrectionRecord:
    """Record of a user correction."""
    
    column_name: str
    predicted_action: PreprocessingAction
    corrected_action: PreprocessingAction
    column_statistics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    dataset_id: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'column_name': self.column_name,
            'predicted_action': self.predicted_action.value if isinstance(self.predicted_action, PreprocessingAction) else self.predicted_action,
            'corrected_action': self.corrected_action.value if isinstance(self.corrected_action, PreprocessingAction) else self.corrected_action,
            'column_statistics': self.column_statistics,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'dataset_id': self.dataset_id,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrectionRecord':
        """Create from dictionary."""
        return cls(
            column_name=data['column_name'],
            predicted_action=PreprocessingAction(data['predicted_action']) if isinstance(data['predicted_action'], str) else data['predicted_action'],
            corrected_action=PreprocessingAction(data['corrected_action']) if isinstance(data['corrected_action'], str) else data['corrected_action'],
            column_statistics=data['column_statistics'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            dataset_id=data.get('dataset_id'),
            confidence=data.get('confidence', 0.0)
        )


@dataclass
class ColumnPattern:
    """Pattern extracted from multiple corrections."""
    
    pattern_id: str
    action: PreprocessingAction
    support: int  # Number of corrections matching this pattern
    confidence: float  # Percentage of corrections with this action
    conditions: Dict[str, Any]  # Statistical conditions defining the pattern
    examples: List[str] = field(default_factory=list)  # Example column names
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Generate pattern ID if not provided."""
        if not self.pattern_id:
            self.pattern_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique pattern ID."""
        content = json.dumps({
            'action': self.action.value if isinstance(self.action, PreprocessingAction) else self.action,
            'conditions': self.conditions
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def matches(self, statistics: Dict[str, Any], threshold: float = 0.8) -> bool:
        """
        Check if column statistics match this pattern.
        
        Args:
            statistics: Column statistics
            threshold: Matching threshold (0-1)
            
        Returns:
            True if statistics match the pattern
        """
        matches = 0
        total = 0
        
        for key, condition in self.conditions.items():
            if key not in statistics:
                continue
            
            total += 1
            stat_value = statistics[key]
            
            if isinstance(condition, dict):
                # Range condition
                if 'min' in condition and 'max' in condition:
                    if condition['min'] <= stat_value <= condition['max']:
                        matches += 1
            elif isinstance(condition, bool):
                # Boolean condition
                if stat_value == condition:
                    matches += 1
            else:
                # Exact match
                if stat_value == condition:
                    matches += 1
        
        if total == 0:
            return False
        
        return (matches / total) >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'action': self.action.value if isinstance(self.action, PreprocessingAction) else self.action,
            'support': self.support,
            'confidence': self.confidence,
            'conditions': self.conditions,
            'examples': self.examples,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'last_seen': self.last_seen.isoformat() if isinstance(self.last_seen, datetime) else self.last_seen
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnPattern':
        """Create from dictionary."""
        return cls(
            pattern_id=data['pattern_id'],
            action=PreprocessingAction(data['action']) if isinstance(data['action'], str) else data['action'],
            support=data['support'],
            confidence=data['confidence'],
            conditions=data['conditions'],
            examples=data.get('examples', []),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at'],
            last_seen=datetime.fromisoformat(data['last_seen']) if isinstance(data['last_seen'], str) else data['last_seen']
        )


class LocalPatternLearner:
    """
    Local pattern learner that extracts patterns from corrections.
    Stores patterns in memory or on disk.
    """
    
    def __init__(
        self,
        min_support: int = 10,
        min_confidence: float = 0.8,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize pattern learner.
        
        Args:
            min_support: Minimum number of corrections to create a pattern
            min_confidence: Minimum confidence for a pattern
            storage_path: Path to store patterns on disk
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.storage_path = storage_path
        
        self.corrections: List[CorrectionRecord] = []
        self.patterns: List[ColumnPattern] = []
        
        # Load patterns from disk if available
        if storage_path and storage_path.exists():
            self.load_patterns()
    
    def record_correction(
        self,
        column_name: str,
        predicted_action: PreprocessingAction,
        corrected_action: PreprocessingAction,
        column_statistics: Dict[str, Any],
        dataset_id: Optional[str] = None,
        confidence: float = 0.0
    ):
        """
        Record a user correction.
        
        Args:
            column_name: Name of the column
            predicted_action: Action predicted by the system
            corrected_action: Action corrected by the user
            column_statistics: Statistics of the column
            dataset_id: Optional dataset identifier
            confidence: Confidence of the prediction
        """
        correction = CorrectionRecord(
            column_name=column_name,
            predicted_action=predicted_action,
            corrected_action=corrected_action,
            column_statistics=column_statistics,
            dataset_id=dataset_id,
            confidence=confidence
        )
        
        self.corrections.append(correction)
        
        # Try to learn patterns when we have enough corrections
        if len(self.corrections) >= self.min_support:
            self._learn_patterns()
    
    def _learn_patterns(self):
        """Learn patterns from accumulated corrections."""
        # Group corrections by action
        action_groups: Dict[PreprocessingAction, List[CorrectionRecord]] = {}
        
        for correction in self.corrections:
            action = correction.corrected_action
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append(correction)
        
        # Extract patterns for each action group
        for action, corrections in action_groups.items():
            if len(corrections) >= self.min_support:
                pattern = self._extract_pattern(action, corrections)
                if pattern and pattern.confidence >= self.min_confidence:
                    # Check if pattern already exists
                    existing = self._find_pattern(pattern.pattern_id)
                    if existing:
                        # Update existing pattern
                        existing.support += pattern.support
                        existing.last_seen = datetime.now()
                        existing.examples.extend(pattern.examples)
                    else:
                        self.patterns.append(pattern)
        
        # Save patterns to disk
        if self.storage_path:
            self.save_patterns()
    
    def _extract_pattern(
        self,
        action: PreprocessingAction,
        corrections: List[CorrectionRecord]
    ) -> Optional[ColumnPattern]:
        """
        Extract a pattern from a group of corrections.
        
        Args:
            action: The corrected action
            corrections: List of corrections with this action
            
        Returns:
            Extracted pattern or None
        """
        if not corrections:
            return None
        
        # Extract common statistical conditions
        conditions = {}
        
        # Analyze numeric statistics
        numeric_keys = ['null_pct', 'unique_ratio', 'skewness', 'kurtosis', 'mean', 'std']
        for key in numeric_keys:
            values = []
            for corr in corrections:
                if key in corr.column_statistics:
                    val = corr.column_statistics[key]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        values.append(val)
            
            if values:
                # Create a range condition
                conditions[key] = {
                    'min': min(values),
                    'max': max(values)
                }
        
        # Analyze boolean statistics
        boolean_keys = ['is_numeric', 'is_categorical', 'is_temporal', 'has_outliers']
        for key in boolean_keys:
            # Check if all corrections have the same boolean value
            values = [corr.column_statistics.get(key) for corr in corrections if key in corr.column_statistics]
            if values and all(v == values[0] for v in values):
                conditions[key] = values[0]
        
        # Calculate confidence
        confidence = len(corrections) / len(self.corrections)
        
        # Create pattern
        pattern = ColumnPattern(
            pattern_id="",  # Will be generated
            action=action,
            support=len(corrections),
            confidence=confidence,
            conditions=conditions,
            examples=[c.column_name for c in corrections[:5]]  # Store up to 5 examples
        )
        
        return pattern
    
    def _find_pattern(self, pattern_id: str) -> Optional[ColumnPattern]:
        """Find a pattern by ID."""
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None
    
    def get_matching_patterns(
        self,
        statistics: Dict[str, Any],
        threshold: float = 0.8
    ) -> List[ColumnPattern]:
        """
        Get patterns that match the given statistics.
        
        Args:
            statistics: Column statistics
            threshold: Matching threshold
            
        Returns:
            List of matching patterns
        """
        matches = []
        for pattern in self.patterns:
            if pattern.matches(statistics, threshold):
                matches.append(pattern)
        return matches
    
    def save_patterns(self):
        """Save patterns to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'patterns': [p.to_dict() for p in self.patterns],
            'corrections': [c.to_dict() for c in self.corrections]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_patterns(self):
        """Load patterns from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        self.patterns = [ColumnPattern.from_dict(p) for p in data.get('patterns', [])]
        self.corrections = [CorrectionRecord.from_dict(c) for c in data.get('corrections', [])]
    
    def clear(self):
        """Clear all corrections and patterns."""
        self.corrections = []
        self.patterns = []
        
        if self.storage_path and self.storage_path.exists():
            self.storage_path.unlink()
