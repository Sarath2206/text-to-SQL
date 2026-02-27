"""
SQL-R1 Extensions Package

This package contains enhancements to the SQL-R1 codebase:
- Enhanced reward computation (schema-aware, structural, syntax)
- 24GB GPU optimization utilities
- Enhanced logging and metrics tracking
- Checkpoint management improvements
"""

__version__ = "0.1.0"

# Import main components for easy access
from .reward_enhanced import (
    EnhancedRewardComputer,
    SchemaAwareReward,
    StructuralReward,
    EnhancedSyntaxReward,
)

from .logging_enhanced import EnhancedLogger

__all__ = [
    "EnhancedRewardComputer",
    "SchemaAwareReward",
    "StructuralReward",
    "EnhancedSyntaxReward",
    "EnhancedLogger",
]
