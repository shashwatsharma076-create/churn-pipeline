"""
Customer Churn Pipeline - Main Package
"""
from src.agents import (
    BaseAgent,
    SignalAgent,
    RiskAgent,
    ActionAgent,
    OrchestratorAgent,
)
from src.schemas import (
    ChurnSignal,
    CustomerProfile,
    RiskAssessment,
    RetentionAction,
    PipelineResult,
)
from src.utils import (
    load_raw_data,
    clean_data,
    load_or_create_clean_data,
    get_data_summary,
    format_assessment,
    save_results,
    print_summary,
    print_top_risk_customers,
)
from src.config import (
    PATHS,
    THRESHOLDS,
    RISK_LEVELS,
    API_CONFIG,
    ensure_directories,
    get_project_root,
)
from src.pipeline import UnifiedChurnPipeline

__version__ = "1.1.0"

__all__ = [
    "BaseAgent",
    "SignalAgent",
    "RiskAgent",
    "ActionAgent",
    "OrchestratorAgent",
    "ChurnSignal",
    "CustomerProfile",
    "RiskAssessment",
    "RetentionAction",
    "PipelineResult",
    "UnifiedChurnPipeline",
    "load_raw_data",
    "clean_data",
    "load_or_create_clean_data",
    "get_data_summary",
    "format_assessment",
    "save_results",
    "print_summary",
    "print_top_risk_customers",
    "PATHS",
    "THRESHOLDS",
    "RISK_LEVELS",
    "API_CONFIG",
    "ensure_directories",
    "get_project_root",
]
