"""
Data schemas for type validation and serialization.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ChurnSignal:
    name: str
    value: float
    threshold: float
    severity: str
    description: str


@dataclass
class CustomerProfile:
    customer_id: int
    age: float
    gender: str
    tenure: float
    usage_frequency: float
    support_calls: float
    payment_delay: float
    subscription_type: str
    contract_length: str
    total_spend: float
    last_interaction: float
    churn: int = 0


@dataclass
class RiskAssessment:
    customer_id: int
    risk_level: str
    risk_score: float
    signal_count: int
    signals: List[ChurnSignal]
    churn_probability: Optional[float] = None
    recommendation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RetentionAction:
    action_type: str
    priority: str
    description: str
    target_customer_segment: str
    expected_impact: str


@dataclass
class PipelineResult:
    total_customers: int
    risk_distribution: dict
    high_risk_count: int
    critical_risk_count: int
    processing_time_seconds: float
    timestamp: str
    results: List[RiskAssessment]
