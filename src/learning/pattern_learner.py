"""
Privacy-Preserving Pattern Learner.
Learns generalizable patterns from user corrections without storing actual data.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

from ..core.actions import PreprocessingAction
from ..symbolic.rules import Rule, RuleCategory
from .privacy import create_privacy_preserving_pattern, AnonymizationUtils


@dataclass
class ColumnPattern:
    """Privacy-preserving pattern extracted from a column."""

    # Statistical signature (discretized, no raw values)
    skew_bucket: Optional[int] = None  # Discretized skewness
    null_bucket: Optional[int] = None  # Discretized null percentage
    cardinality_type: Optional[str] = None  # 'low', 'medium', 'high', 'unique'

    # Type information
    is_numeric: bool = False
    is_categorical: bool = False
    is_temporal: bool = False

    # Semantic patterns (no actual values)
    has_date_pattern: bool = False
    has_currency_pattern: bool = False
    has_email_pattern: bool = False
    has_phone_pattern: bool = False

    # Column name tokens (generalized)
    name_tokens: List[str] = field(default_factory=list)

    # Metadata
    pattern_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'skew_bucket': self.skew_bucket,
            'null_bucket': self.null_bucket,
            'cardinality_type': self.cardinality_type,
            'is_numeric': self.is_numeric,
            'is_categorical': self.is_categorical,
            'is_temporal': self.is_temporal,
            'has_date_pattern': self.has_date_pattern,
            'has_currency_pattern': self.has_currency_pattern,
            'has_email_pattern': self.has_email_pattern,
            'has_phone_pattern': self.has_phone_pattern,
            'name_tokens': self.name_tokens,
            'pattern_hash': self.pattern_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnPattern':
        """Create from dictionary."""
        return cls(**data)

    def similarity(self, other: 'ColumnPattern') -> float:
        """
        Calculate similarity to another pattern.

        Returns:
            Similarity score (0.0 to 1.0)
        """
        score = 0.0
        total_weight = 0.0

        # Statistical similarity
        if self.skew_bucket is not None and other.skew_bucket is not None:
            bucket_diff = abs(self.skew_bucket - other.skew_bucket)
            score += (1.0 - bucket_diff / 5.0) * 2.0  # Weight: 2.0
            total_weight += 2.0

        if self.null_bucket is not None and other.null_bucket is not None:
            bucket_diff = abs(self.null_bucket - other.null_bucket)
            score += (1.0 - bucket_diff / 4.0) * 1.5  # Weight: 1.5
            total_weight += 1.5

        if self.cardinality_type and other.cardinality_type:
            if self.cardinality_type == other.cardinality_type:
                score += 2.0  # Weight: 2.0
            total_weight += 2.0

        # Type similarity
        type_match = sum([
            self.is_numeric == other.is_numeric,
            self.is_categorical == other.is_categorical,
            self.is_temporal == other.is_temporal
        ])
        score += type_match * 1.0  # Weight: 1.0 per type
        total_weight += 3.0

        # Pattern similarity
        pattern_match = sum([
            self.has_date_pattern == other.has_date_pattern,
            self.has_currency_pattern == other.has_currency_pattern,
            self.has_email_pattern == other.has_email_pattern,
            self.has_phone_pattern == other.has_phone_pattern
        ])
        score += pattern_match * 0.5  # Weight: 0.5 per pattern
        total_weight += 2.0

        # Name token similarity
        if self.name_tokens and other.name_tokens:
            common_tokens = set(self.name_tokens) & set(other.name_tokens)
            union_tokens = set(self.name_tokens) | set(other.name_tokens)
            if union_tokens:
                jaccard = len(common_tokens) / len(union_tokens)
                score += jaccard * 2.0  # Weight: 2.0
            total_weight += 2.0

        if total_weight == 0:
            return 0.0

        return score / total_weight


@dataclass
class CorrectionRecord:
    """Record of a user correction (privacy-preserving)."""
    pattern: ColumnPattern
    wrong_action: PreprocessingAction
    correct_action: PreprocessingAction
    timestamp: float
    confidence_score: float = 0.0  # How confident the system was


class LocalPatternLearner:
    """
    Learns generalizable patterns from corrections locally.
    Never stores actual data values - only statistical patterns.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_pattern_support: int = 3,
        privacy_level: str = "high"
    ):
        """
        Initialize the pattern learner.

        Args:
            similarity_threshold: Minimum similarity to consider patterns related
            min_pattern_support: Minimum occurrences to generalize into rule
            privacy_level: 'low', 'medium', or 'high' privacy
        """
        self.similarity_threshold = similarity_threshold
        self.min_pattern_support = min_pattern_support
        self.privacy_level = privacy_level

        # Pattern memory (privacy-preserving)
        self.correction_records: List[CorrectionRecord] = []

        # Learned rules
        self.learned_rules: List[Rule] = []

        # Pattern clusters
        self.pattern_clusters: Dict[str, List[CorrectionRecord]] = defaultdict(list)

    def extract_pattern(
        self,
        column_stats: Dict[str, Any],
        column_name: str = ""
    ) -> ColumnPattern:
        """
        Extract privacy-preserving pattern from column statistics.

        Args:
            column_stats: Column statistics from symbolic engine
            column_name: Name of the column

        Returns:
            ColumnPattern with no actual data values
        """
        anonymizer = AnonymizationUtils()

        # Discretize skewness
        skew_bucket = None
        if column_stats.get('skewness') is not None:
            skew_bucket = anonymizer.discretize_value(
                column_stats['skewness'],
                bins=5,
                value_range=(-3, 3)
            )

        # Discretize null percentage
        null_bucket = None
        if column_stats.get('null_pct') is not None:
            null_bucket = anonymizer.discretize_value(
                column_stats['null_pct'],
                bins=4,
                value_range=(0, 1)
            )

        # Classify cardinality
        cardinality_type = None
        if column_stats.get('unique_ratio') is not None:
            ratio = column_stats['unique_ratio']
            if ratio > 0.95:
                cardinality_type = 'unique'
            elif ratio > 0.5:
                cardinality_type = 'high'
            elif ratio > 0.1:
                cardinality_type = 'medium'
            else:
                cardinality_type = 'low'

        # Extract name tokens (generalized)
        name_tokens = self._tokenize_column_name(column_name)

        # Create pattern
        pattern = ColumnPattern(
            skew_bucket=skew_bucket,
            null_bucket=null_bucket,
            cardinality_type=cardinality_type,
            is_numeric=column_stats.get('is_numeric', False),
            is_categorical=column_stats.get('is_categorical', False),
            is_temporal=column_stats.get('is_temporal', False),
            has_date_pattern=column_stats.get('matches_date_pattern', 0) > 0.5,
            has_currency_pattern=column_stats.get('has_currency_symbols', False),
            has_email_pattern=column_stats.get('matches_email_pattern', 0) > 0.5,
            has_phone_pattern=column_stats.get('matches_phone_pattern', 0) > 0.5,
            name_tokens=name_tokens
        )

        # Compute pattern hash for clustering
        pattern.pattern_hash = self._compute_pattern_hash(pattern)

        return pattern

    def _tokenize_column_name(self, name: str) -> List[str]:
        """Tokenize column name into meaningful parts."""
        import re

        if not name:
            return []

        # Split on common separators and camelCase
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+', name)

        # Convert to lowercase and filter common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at'}
        tokens = [t.lower() for t in tokens if t.lower() not in stopwords]

        return tokens

    def _compute_pattern_hash(self, pattern: ColumnPattern) -> str:
        """Compute hash for pattern clustering."""
        # Create a string representation of key pattern features
        pattern_str = f"{pattern.is_numeric}_{pattern.is_categorical}_" \
                     f"{pattern.cardinality_type}_{pattern.null_bucket}"

        return AnonymizationUtils.hash_value(pattern_str)[:8]

    def learn_correction(
        self,
        pattern: ColumnPattern,
        wrong_action: PreprocessingAction,
        correct_action: PreprocessingAction,
        confidence: float = 0.0
    ) -> Optional[Rule]:
        """
        Learn from a user correction.

        Args:
            pattern: Privacy-preserving pattern
            wrong_action: Action that was incorrect
            correct_action: Correct action
            confidence: Confidence of wrong prediction

        Returns:
            New learned rule if pattern generalizes, None otherwise
        """
        # Record the correction
        record = CorrectionRecord(
            pattern=pattern,
            wrong_action=wrong_action,
            correct_action=correct_action,
            timestamp=np.random.random(),  # Would use time.time() in production
            confidence_score=confidence
        )

        self.correction_records.append(record)

        # Add to cluster
        if pattern.pattern_hash:
            self.pattern_clusters[pattern.pattern_hash].append(record)

        # Check if we can generalize into a rule
        similar_patterns = self.find_similar_patterns(pattern, self.similarity_threshold)

        if len(similar_patterns) >= self.min_pattern_support:
            # Check if they all have the same correct action
            actions = [r.correct_action for r in similar_patterns]
            most_common_action = max(set(actions), key=actions.count)

            if actions.count(most_common_action) >= self.min_pattern_support:
                # Create a new rule
                new_rule = self._generalize_patterns(similar_patterns, most_common_action)
                self.learned_rules.append(new_rule)
                return new_rule

        return None

    def find_similar_patterns(
        self,
        pattern: ColumnPattern,
        threshold: float
    ) -> List[CorrectionRecord]:
        """
        Find correction records with similar patterns.

        Args:
            pattern: Pattern to match
            threshold: Minimum similarity threshold

        Returns:
            List of similar correction records
        """
        similar = []

        for record in self.correction_records:
            similarity = pattern.similarity(record.pattern)
            if similarity >= threshold:
                similar.append(record)

        return similar

    def _generalize_patterns(
        self,
        records: List[CorrectionRecord],
        action: PreprocessingAction
    ) -> Rule:
        """
        Generalize multiple patterns into a single rule.

        Args:
            records: Correction records to generalize
            action: The action this rule should recommend

        Returns:
            Generalized rule
        """
        # Find common characteristics
        patterns = [r.pattern for r in records]

        # Common type
        is_numeric = all(p.is_numeric for p in patterns)
        is_categorical = all(p.is_categorical for p in patterns)

        # Common cardinality
        cardinality_types = [p.cardinality_type for p in patterns if p.cardinality_type]
        common_cardinality = max(set(cardinality_types), key=cardinality_types.count) \
            if cardinality_types else None

        # Common name tokens
        all_tokens = [token for p in patterns for token in p.name_tokens]
        common_tokens = []
        if all_tokens:
            from collections import Counter
            token_counts = Counter(all_tokens)
            common_tokens = [t for t, c in token_counts.items() if c >= len(patterns) * 0.5]

        # Create condition function
        def condition(stats: Dict[str, Any]) -> bool:
            matches = True

            if is_numeric and not stats.get('is_numeric', False):
                return False
            if is_categorical and not stats.get('is_categorical', False):
                return False

            if common_cardinality:
                ratio = stats.get('unique_ratio', 0)
                actual_type = (
                    'unique' if ratio > 0.95
                    else 'high' if ratio > 0.5
                    else 'medium' if ratio > 0.1
                    else 'low'
                )
                if actual_type != common_cardinality:
                    return False

            return matches

        # Calculate confidence based on pattern support
        support = len(records)
        confidence = min(0.9, 0.6 + (support - self.min_pattern_support) * 0.1)

        # Create rule
        rule_name = f"LEARNED_{action.value.upper()}_{len(self.learned_rules)}"

        rule = Rule(
            name=rule_name,
            category=RuleCategory.DATA_QUALITY,
            action=action,
            condition=condition,
            confidence_fn=lambda stats: confidence,
            explanation_fn=lambda stats: f"Learned from {support} similar user corrections",
            priority=85  # High priority for learned rules
        )

        return rule

    def check_patterns(self, column_stats: Dict[str, Any]) -> Optional[PreprocessingAction]:
        """
        Check if any learned patterns match.

        Args:
            column_stats: Column statistics

        Returns:
            Recommended action if pattern matches, None otherwise
        """
        # Extract pattern
        pattern = self.extract_pattern(column_stats, column_stats.get('column_name', ''))

        # Check against learned rules
        for rule in self.learned_rules:
            if rule.condition(column_stats):
                return rule.action

        return None

    def save(self, path: Path):
        """
        Save learned patterns to disk.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for serialization
        data = {
            'correction_records': [
                {
                    'pattern': r.pattern.to_dict(),
                    'wrong_action': r.wrong_action.value,
                    'correct_action': r.correct_action.value,
                    'timestamp': r.timestamp,
                    'confidence_score': r.confidence_score
                }
                for r in self.correction_records
            ],
            'learned_rules_count': len(self.learned_rules),
            'similarity_threshold': self.similarity_threshold,
            'min_pattern_support': self.min_pattern_support,
            'privacy_level': self.privacy_level
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        """
        Load learned patterns from disk.

        Args:
            path: Path to load from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pattern file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Restore correction records
        self.correction_records = []
        for record_data in data['correction_records']:
            pattern = ColumnPattern.from_dict(record_data['pattern'])
            record = CorrectionRecord(
                pattern=pattern,
                wrong_action=PreprocessingAction(record_data['wrong_action']),
                correct_action=PreprocessingAction(record_data['correct_action']),
                timestamp=record_data['timestamp'],
                confidence_score=record_data['confidence_score']
            )
            self.correction_records.append(record)

            # Rebuild clusters
            if pattern.pattern_hash:
                self.pattern_clusters[pattern.pattern_hash].append(record)

        # Rebuild learned rules
        self._rebuild_rules()

        # Restore settings
        self.similarity_threshold = data.get('similarity_threshold', 0.8)
        self.min_pattern_support = data.get('min_pattern_support', 3)
        self.privacy_level = data.get('privacy_level', 'high')

    def _rebuild_rules(self):
        """Rebuild learned rules from correction records."""
        self.learned_rules = []

        # Group by pattern hash
        for cluster in self.pattern_clusters.values():
            if len(cluster) >= self.min_pattern_support:
                # Find most common correct action
                actions = [r.correct_action for r in cluster]
                most_common = max(set(actions), key=actions.count)

                if actions.count(most_common) >= self.min_pattern_support:
                    rule = self._generalize_patterns(cluster, most_common)
                    self.learned_rules.append(rule)

    def get_statistics(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            'total_corrections': len(self.correction_records),
            'learned_rules': len(self.learned_rules),
            'pattern_clusters': len(self.pattern_clusters),
            'avg_cluster_size': np.mean([len(c) for c in self.pattern_clusters.values()])
                if self.pattern_clusters else 0
        }
