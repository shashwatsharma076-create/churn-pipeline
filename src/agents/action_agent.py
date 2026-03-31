"""
Action Agent - Generates retention recommendations.
"""
from typing import List
from .base import BaseAgent
from src.schemas import RiskAssessment, RetentionAction


SYSTEM_PROMPT = """You are a customer retention strategist.
Given customer risk assessments, recommend specific actions to prevent churn.
Be actionable, specific, and prioritize based on risk level."""


class ActionAgent(BaseAgent):
    """Agent responsible for generating retention actions."""

    def run(self, assessment: RiskAssessment) -> List[RetentionAction]:
        """Generate retention actions based on risk assessment."""
        actions = []

        if assessment.risk_level in ["critical", "high"]:
            actions.extend(self._generate_urgent_actions(assessment))
        elif assessment.risk_level == "medium":
            actions.extend(self._generate_preventive_actions(assessment))
        else:
            actions.extend(self._generate_loyalty_actions(assessment))

        for action in actions:
            action.description = self._enhance_with_llm(action, assessment)

        return actions

    def _generate_urgent_actions(self, assessment: RiskAssessment) -> List[RetentionAction]:
        """Generate urgent retention actions for high/critical risk."""
        actions = []
        signal_names = [s.name for s in assessment.signals]

        if "payment_delay" in signal_names:
            actions.append(RetentionAction(
                action_type="payment_plan",
                priority="high",
                description="Offer payment plan or grace period",
                target_customer_segment="payment_concern",
                expected_impact="Reduce churn by addressing financial friction"
            ))

        if "support_calls" in signal_names:
            actions.append(RetentionAction(
                action_type="proactive_support",
                priority="high",
                description="Assign dedicated support representative",
                target_customer_segment="support_concern",
                expected_impact="Improve satisfaction and reduce frustration"
            ))

        if "last_interaction" in signal_names:
            actions.append(RetentionAction(
                action_type="re_engagement",
                priority="high",
                description="Send personalized re-engagement campaign",
                target_customer_segment="inactive",
                expected_impact="Reconnect with disengaged customer"
            ))

        if "usage_frequency" in signal_names:
            actions.append(RetentionAction(
                action_type="usage_education",
                priority="medium",
                description="Provide onboarding or feature tutorial",
                target_customer_segment="low_engagement",
                expected_impact="Increase product adoption"
            ))

        if "total_spend" in signal_names:
            actions.append(RetentionAction(
                action_type="incentive_offer",
                priority="medium",
                description="Offer exclusive discount or upgrade",
                target_customer_segment="value_concern",
                expected_impact="Increase perceived value"
            ))

        actions.append(RetentionAction(
            action_type="direct_outreach",
            priority="critical" if assessment.risk_level == "critical" else "high",
            description="Schedule direct call with customer success manager",
            target_customer_segment="all_at_risk",
            expected_impact="Personal connection to understand concerns"
        ))

        return actions

    def _generate_preventive_actions(self, assessment: RiskAssessment) -> List[RetentionAction]:
        """Generate preventive actions for medium risk."""
        return [
            RetentionAction(
                action_type="check_in",
                priority="medium",
                description="Send satisfaction survey and check-in email",
                target_customer_segment="monitoring",
                expected_impact="Early warning system for emerging issues"
            ),
            RetentionAction(
                action_type="feature_highlight",
                priority="low",
                description="Share relevant feature updates and tips",
                target_customer_segment="engagement",
                expected_impact="Increase product value perception"
            ),
        ]

    def _generate_loyalty_actions(self, assessment: RiskAssessment) -> List[RetentionAction]:
        """Generate loyalty actions for low/no risk."""
        return [
            RetentionAction(
                action_type="appreciation",
                priority="low",
                description="Send thank you and loyalty reward",
                target_customer_segment="loyal",
                expected_impact="Strengthen customer loyalty"
            ),
            RetentionAction(
                action_type="referral",
                priority="low",
                description="Invite to referral program",
                target_customer_segment="advocates",
                expected_impact="Turn loyal customers into advocates"
            ),
        ]

    def _enhance_with_llm(self, action: RetentionAction, assessment: RiskAssessment) -> str:
        """Enhance action description using LLM."""
        prompt = f"""Customer Risk Level: {assessment.risk_level.upper()}
        Risk Score: {assessment.risk_score}/100
        Signal Count: {assessment.signal_count}

        Action Type: {action.action_type}
        Base Description: {action.description}

        Provide a more specific, personalized version of this action description 
        (1-2 sentences max). Focus on concrete next steps."""
        return self.call_llm(prompt, SYSTEM_PROMPT)

    def generate_email_content(self, assessment: RiskAssessment, action: RetentionAction) -> str:
        """Generate personalized retention email content."""
        signal_summary = ", ".join([s.description for s in assessment.signals[:2]])
        prompt = f"""Write a personalized retention email for a customer with:
        - Risk Level: {assessment.risk_level}
        - Key Concerns: {signal_summary}
        - Recommended Action: {action.description}

        The email should be warm, not pushy, and focus on addressing their specific concerns.
        Keep it under 150 words. No subject line needed."""
        return self.call_llm(prompt, SYSTEM_PROMPT)
