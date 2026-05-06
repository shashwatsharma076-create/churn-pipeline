"""
Email templates for different risk levels.
"""
from typing import Dict


# Email templates for different risk levels
EMAIL_TEMPLATES = {
    "critical": {
        "subject": "Urgent: Personal outreach regarding your account",
        "template": """Dear {customer_name},

We noticed some concerning patterns with your account recently:

{risk_signals}

We'd like to schedule a personal call with our Customer Success team to address these issues and ensure you're getting the most value from our service.

As a valued customer, we're prepared to offer:
- Priority support (no wait times)
- A dedicated account manager
- Special retention pricing for your next renewal

Would you be available for a brief 10-minute call this week? Simply reply to this email with your availability.

Best regards,
Customer Success Team
{company_name}

P.S. We truly value your business and want to make things right.""",
    },
    "high": {
        "subject": "We value your business - Special offer inside",
        "template": """Dear {customer_name},

Thank you for being a customer! We've noticed a few things that suggest you might not be getting the full value from your subscription:

{risk_signals}

To help you get back on track, we'd like to offer:
- 20% discount on your next renewal
- Free upgrade to Premium for 1 month
- A complimentary account review with our support team

Click here to claim your offer: [Claim Now]

If you have any questions or concerns, simply reply to this email - we're here to help.

Best,
Customer Success Team
{company_name}""",
    },
    "medium": {
        "subject": "Tips to get more from your subscription",
        "template": """Dear {customer_name},

We hope you're enjoying our service! We noticed a few areas where you might benefit from additional support:

{risk_signals}

Here are some tips to maximize your experience:
- Check out our knowledge base for quick answers
- Join our community forum to connect with other users
- Watch our tutorial videos for advanced features

We also have a quick 2-minute satisfaction survey: [Take Survey]

Thanks for being a customer!

Best,
Customer Success Team
{company_name}""",
    },
    "low": {
        "subject": "Thank you for being a loyal customer!",
        "template": """Dear {customer_name},

We just wanted to say thank you for being a valued customer! Your loyalty means the world to us.

As a token of our appreciation, here's what you have access to:
- Our premium support channel
- Exclusive webinars and training sessions
- Early access to new features

Keep an eye on your inbox for more exciting updates!

Best,
Customer Success Team
{company_name}""",
    },
}

# LLM prompt templates for generating personalized emails
LLM_EMAIL_PROMPTS = {
    "critical": """Generate a personalized retention email for a CRITICAL risk customer with the following details:
- Customer ID: {customer_id}
- Risk Score: {risk_score}/100
- Detected Signals: {signals}
- Churn Probability: {churn_probability:.1%}

The email should:
1. Be empathetic and understanding
2. Address specific issues (support calls, payment delays, etc.)
3. Offer concrete solutions (personal call, special pricing, dedicated support)
4. Include a clear call-to-action
5. Be professional but warm

Return the email in this format:
Subject: <subject>
Body:
<body>""",
    
    "high": """Generate a personalized retention email for a HIGH risk customer with the following details:
- Customer ID: {customer_id}
- Risk Score: {risk_score}/100
- Detected Signals: {signals}
- Churn Probability: {churn_probability:.1%}

The email should:
1. Be friendly and appreciative
2. Mention specific pain points from signals
3. Offer incentives (discounts, upgrades, free trials)
4. Include a clear call-to-action
5. Keep it concise and engaging

Return the email in this format:
Subject: <subject>
Body:
<body>""",
}


def get_template(risk_level: str) -> Dict[str, str]:
    """Get email template for a given risk level."""
    risk_level = risk_level.lower()
    if risk_level not in EMAIL_TEMPLATES:
        # Default to medium if unknown
        risk_level = "medium"
    return EMAIL_TEMPLATES[risk_level]


def format_signals_for_email(signals: list) -> str:
    """Format detected signals into a readable string for emails."""
    if not signals:
        return "- No major issues detected"
    
    formatted = []
    for sig in signals:
        formatted.append(f"- {sig.description} (Severity: {sig.severity})")
    
    return "\n".join(formatted)


def get_llm_prompt(risk_level: str, customer_id: int, risk_score: int, 
                   signals: list, churn_probability: float) -> str:
    """Get LLM prompt for email generation."""
    risk_level = risk_level.lower()
    signals_text = format_signals_for_email(signals)
    
    if risk_level in LLM_EMAIL_PROMPTS:
        return LLM_EMAIL_PROMPTS[risk_level].format(
            customer_id=customer_id,
            risk_score=risk_score,
            signals=signals_text,
            churn_probability=churn_probability
        )
    return ""
