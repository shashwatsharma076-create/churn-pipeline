"""
Email generator for customer churn retention.
Supports template-based and LLM-powered email generation.
"""
import os
import smtplib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.utils.email_templates import (
    get_template, format_signals_for_email, get_llm_prompt
)
from src.config import PATHS


class EmailGenerator:
    """Generate and send retention emails to at-risk customers."""
    
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.llm_client = None
        
        if use_llm:
            try:
                import openai
                api_key = os.getenv("OPENROUTER_API_KEY")
                if api_key:
                    self.llm_client = openai.OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
            except ImportError:
                print("openai package not installed for LLM emails")
    
    def generate_email(self, customer_data: dict, assessment, signals: list) -> Dict[str, str]:
        """
        Generate a personalized email for a customer.
        
        Args:
            customer_data: Customer details dict
            assessment: Risk assessment object (has risk_level, risk_score, churn_probability)
            signals: List of detected signal objects
        
        Returns:
            Dict with 'subject' and 'body' keys
        """
        risk_level = assessment.risk_level.lower()
        
        # Try LLM generation first if enabled
        if self.use_llm and self.llm_client:
            try:
                return self._generate_llm_email(risk_level, customer_data, assessment, signals)
            except Exception as e:
                print(f"LLM generation failed: {e}. Falling back to template.")
        
        # Fall back to template
        return self._generate_template_email(customer_data, assessment, signals)
    
    def _generate_template_email(self, customer_data: dict, assessment, signals: list) -> Dict[str, str]:
        """Generate email using templates."""
        risk_level = assessment.risk_level.lower()
        template_data = get_template(risk_level)
        
        # Format signals
        signals_text = format_signals_for_email(signals)
        
        # Prepare variables
        variables = {
            "customer_name": f"Customer #{customer_data.get('CustomerID', 'Valued')}",
            "risk_signals": signals_text,
            "company_name": "Your Company",
            "risk_score": assessment.risk_score,
            "churn_probability": f"{assessment.churn_probability:.1%}"
        }
        
        # Fill template
        subject = template_data["subject"]
        body = template_data["template"].format(**variables)
        
        return {"subject": subject, "body": body}
    
    def _generate_llm_email(self, risk_level: str, customer_data: dict, 
                            assessment, signals: list) -> Dict[str, str]:
        """Generate email using LLM."""
        customer_id = customer_data.get("CustomerID", 0)
        prompt = get_llm_prompt(
            risk_level, customer_id, assessment.risk_score, 
            signals, assessment.churn_probability
        )
        
        response = self.llm_client.chat.completions.create(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Parse response
        subject = ""
        body = ""
        if "Subject:" in content:
            parts = content.split("Subject:", 1)
            rest = parts[1]
            if "Body:" in rest:
                subj, body_part = rest.split("Body:", 1)
                subject = subj.strip()
                body = body_part.strip()
            else:
                subject = rest.strip()
        else:
            body = content
        
        return {"subject": subject or "Important message about your account", "body": body}
    
    def send_email(self, to_email: str, subject: str, body: str, 
                   from_email: str = None, from_password: str = None) -> bool:
        """
        Send an email using SMTP.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body
            from_email: Sender email (or from env)
            from_password: Sender password (or from env)
        
        Returns:
            True if sent successfully, False otherwise
        """
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        from_email = from_email or os.getenv("SMTP_USERNAME")
        from_password = from_password or os.getenv("SMTP_PASSWORD")
        
        if not from_email or not from_password:
            print("SMTP credentials not configured. Email not sent.")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(from_email, from_password)
            server.send_message(msg)
            server.quit()
            
            print(f"Email sent to {to_email}")
            return True
        except Exception as e:
            print(f"Failed to send email to {to_email}: {e}")
            return False
    
    def generate_bulk_emails(self, customers: list, assessments: list, 
                            all_signals: list) -> List[Dict]:
        """
        Generate emails for multiple customers.
        
        Args:
            customers: List of customer data dicts
            assessments: List of assessment objects
            all_signals: List of signal lists for each customer
        
        Returns:
            List of dicts with keys: customer_id, email, subject, body, status
        """
        results = []
        
        for i, customer in enumerate(customers):
            assessment = assessments[i]
            signals = all_signals[i] if i < len(all_signals) else []
            
            # Skip low/none risk unless explicitly included
            if assessment.risk_level.lower() in ["low", "none"]:
                continue
            
            email_data = self.generate_email(customer, assessment, signals)
            
            results.append({
                "customer_id": customer.get("CustomerID", i),
                "email": customer.get("Email", f"customer_{customer.get('CustomerID', i)}@example.com"),
                "risk_level": assessment.risk_level.upper(),
                "risk_score": assessment.risk_score,
                "subject": email_data["subject"],
                "body": email_data["body"],
                "status": "Generated"
            })
        
        return results
    
    def export_to_csv(self, email_data: List[Dict], output_path: Path = None) -> Path:
        """Export generated emails to CSV."""
        if output_path is None:
            output_path = PATHS["outputs"] / "retention_emails.csv"
        
        df = pd.DataFrame(email_data)
        df.to_csv(output_path, index=False)
        print(f"Emails exported to: {output_path}")
        return output_path
