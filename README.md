# Customer Churn Prediction Pipeline

An end-to-end customer churn analysis pipeline using **Agentic AI** - featuring specialized AI agents that collaborate to detect churn signals, assess risk, and recommend retention actions.....

## Architecture

The pipeline uses a multi-agent architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Orchestrator Agent                               │
│                  (Coordinates the entire pipeline)                      │
│                                                                          │
│    1. Receives customer data                                             │
│    2. Routes to appropriate agents                                       │
│    3. Aggregates results                                                 │
│    4. Generates final assessment                                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Signal Agent     │    │   Risk Agent     │    │  Action Agent    │
│                   │    │                  │    │                  │
│ Detects behavioral│    │ Assesses overall │    │ Generates        │
│ patterns that     │    │ churn probability│    │ retention        │
│ indicate churn    │    │ and risk level   │    │ recommendations  │
│ risk from         │    │ based on        │    │ based on risk   │
│ customer metrics  │    │ signals + ML    │    │ assessment      │
└───────────────────┘    └──────────────────┘    └──────────────────┘
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Unified Output        │
                    │  - Risk Assessment     │
                    │  - Detected Signals    │
                    │  - Recommended Actions │
                    └────────────────────────┘
```

## Semantic Workflow Pipeline

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Raw Data   │────▶│ Data Loader  │────▶│   Preprocess  │────▶│ ML Training │
│  (CSV/JSON) │     │  (Clean &    │     │ (Encode,      │     │ (Optional)  │
│             │     │   Validate)  │     │  Scale)       │     │             │
└─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Final      │◀────│ Orchestrator │◀────│ Risk Agent    │◀────│Signal Agent │
│  Output     │     │ (Coordinate) │     │ (Assess Risk) │     │(Detect)     │
│             │     └──────────────┘     └───────────────┘     └─────────────┘
└─────────────┘
```

### Pipeline Stages

#### Stage 1: Data Ingestion
- Load raw customer data from CSV
- Validate data integrity
- Handle missing values
- Clean and normalize

#### Stage 2: Signal Detection (Signal Agent)
Analyzes customer metrics against configurable thresholds:

| Metric | High Threshold | Critical Threshold |
|--------|---------------|-------------------|
| Support Calls | ≥4 | ≥7 |
| Payment Delay | ≥17 days | ≥25 days |
| Last Interaction | ≥15 days | ≥25 days |
| Total Spend | ≤$600 | ≤$300 |
| Usage Frequency | ≤14 days/mo | ≤7 days/mo |
| Tenure | ≤6 months | ≤3 months |

**Severity Levels:**
- `critical` - Immediate intervention required
- `high` - Proactive outreach needed
- `medium` - Monitor closely
- `low` - Standard engagement

#### Stage 3: Risk Assessment (Risk Agent)
Calculates overall risk score:
- **Critical signal**: +25 points
- **High signal**: +15 points
- **Medium signal**: +5 points

Risk Levels:
- 0-9: None
- 10-24: Low
- 25-49: Medium
- 50-74: High
- 75-100: Critical

Contract adjustments: Annual contracts reduce churn probability by 30%, Quarterly by 15%

#### Stage 4: Action Recommendations (Action Agent)
Generates retention actions based on risk level:

- **Critical**: Personal outreach, special offers, executive involvement
- **High**: Proactive support, loyalty rewards, check-in calls
- **Medium**: Engagement campaigns, satisfaction surveys
- **Low**: Loyalty programs, regular communications

#### Stage 5: Orchestration & Output
The Orchestrator coordinates all agents and produces:
- Risk assessment per customer
- Detected signals with severity
- Recommended retention actions
- Risk distribution summary

### Unified Pipeline (ML + AI)

Combines machine learning predictions with AI agent analysis:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Customer   │────▶│  ML Churn    │────▶│  Risk Filter  │
│  Data       │     │  Predictor   │     │  (threshold)  │
└─────────────┘     └──────────────┘     └───────────────┘
                                                  │
                                                  ▼
                                         ┌───────────────┐
                                         │  AI Agent     │
                                         │  Deep Analysis│
                                         └───────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env

# Run the analysis
python main.py analyze

# Run with LLM enhancement for better insights
python main.py analyze --llm

# Analyze a sample
python main.py analyze --sample 100 --llm

# Run unified pipeline (ML + AI)
python main.py unified --sample 1000

# Show data summary
python main.py summary
```

## Features

- **Signal Detection**: Identifies behavioral patterns indicating churn risk
- **Risk Assessment**: Calculates risk scores and probability
- **Retention Actions**: Generates personalized recommendations
- **LLM Enhancement**: Uses AI for deeper semantic analysis
- **ML Integration**: Optional machine learning predictions
- **Unified Pipeline**: Combines ML + AI agents
- **Interactive UI**: Streamlit dashboard with visualizations, CSV upload, and real-time predictions
- **Email Campaign**: Generate and send personalized retention emails to at-risk customers (template or LLM-powered)

## Interactive UI (Streamlit)

The project includes a beautiful, interactive web dashboard built with Streamlit and Plotly.

### Features
- **📊 Dashboard**: Dataset overview with interactive charts (pie, histogram, box plots, correlation heatmap)
- **👤 Single Customer**: Real-time churn prediction with risk gauge and AI recommendations
- **📦 Batch Analysis**: Analyze thousands of customers at once with risk distribution charts
- **🤖 ML Model**: Train models and make predictions with confidence gauges
- **📁 Upload CSV**: Upload your own dataset with auto-detection and template download
- **📧 Email Campaign**: Generate and send personalized retention emails to at-risk customers
- **📥 Export**: Download analysis results as CSV (including emails)

### Screenshots

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    📉 Customer Churn Analyzer                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [📊 Dashboard]  [👤 Single Customer]  [📦 Batch Analysis]           │
│  [🤖 ML Model]  [📁 Upload CSV]                                        │
│                                                                          │
│  ──────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  📊 DATASET OVERVIEW                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │Total:    │  │Churned:  │  │Rate:     │  │Avg:      │              │
│  │440,832   │  │250,000  │  │56.7%     │  │24 mo     │              │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘              │
│                                                                          │
│  [Churn Pie Chart]              [Age Distribution Histogram]             │
│                                                                          │
│  [Feature Correlation Heatmap]                                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Run the UI

```bash
# Install dependencies (including UI packages)
pip install -r requirements.txt

# Launch the dashboard
streamlit run streamlit_app.py
```

The app will open automatically at `http://localhost:8501`

### UI Pages

| Page | Description |
|------|-------------|
| **📊 Dashboard** | Overview of dataset with interactive Plotly charts |
| **👤 Single Customer** | Enter customer details, get AI risk assessment + recommendations |
| **📦 Batch Analysis** | Analyze up to 10,000 customers with risk distribution |
| **🤖 ML Model** | Train models (LR, RF, GB) and make predictions |
| **📁 Upload CSV** | Upload custom datasets with auto-column mapping |

## Project Structure

```
churn-pipeline/
├── streamlit_app.py   # Interactive Streamlit UI
├── src/
│   ├── agents/          # AI agents (signal, risk, action, orchestrator)
│   ├── models/          # ML models (trainer, predictor)
│   ├── pipeline/       # Unified pipeline
│   ├── schemas/        # Data models and types
│   ├── utils/         # Utilities (data loading, output formatting)
│   └── config.py      # Configuration and thresholds
├── data/
│   ├── raw/           # Raw customer data
│   ├── processed/     # Cleaned data
│   └── simulated/     # Generated content
├── outputs/           # Analysis results
├── tests/             # Unit tests
├── main.py            # CLI entry point
└── requirements.txt
```
churn-pipeline/
├── src/
│   ├── agents/          # AI agents (signal, risk, action, orchestrator)
│   ├── models/          # ML models (trainer, predictor)
│   ├── pipeline/       # Unified pipeline
│   ├── schemas/        # Data models and types
│   ├── utils/         # Utilities (data loading, output formatting)
│   └── config.py      # Configuration and thresholds
├── data/
│   ├── raw/           # Raw customer data
│   ├── processed/     # Cleaned data
│   └── simulated/    # Generated content
├── outputs/           # Analysis results
├── tests/             # Unit tests
├── main.py            # CLI entry point
└── requirements.txt
```

## Usage Examples

### Analyze all customers
```bash
python main.py analyze
```

### Analyze with LLM insights
```bash
python main.py analyze --llm
```

### Unified pipeline (ML + AI)
```bash
python main.py unified --sample 1000 --threshold 0.3
```

### Single customer analysis
```bash
python main.py single --customer '{"Age": 35, "Tenure": 12, "Support Calls": 5, ...}'
```

### Show data summary
```bash
python main.py summary
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_signal_agent.py -v
```

## Email Campaign Feature

Generate and send personalized retention emails to at-risk customers.

### How It Works

1. **Load At-Risk Customers**: Filter by Critical/High risk levels
2. **Generate Emails**: Choose template-based or LLM-powered generation
3. **Preview & Edit**: Review emails in the UI before sending
4. **Send or Export**: Send via SMTP (if configured) or download as CSV

### Email Generation Methods

| Method | Description |
|--------|-------------|
| **Template** | Pre-written templates for each risk level (Critical, High, Medium, Low) with customer variables |
| **LLM-Powered** | Uses OpenRouter API with Anthropic Claude 3 Haiku for personalized emails |

### SMTP Configuration (Optional)

To send real emails, configure `.env`:
```bash
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

**Note**: If SMTP is not configured, the app simulates sending and allows CSV export for manual sending.

### Files Added

| File | Purpose |
|------|---------|
| `src/utils/email_templates.py` | Email templates and LLM prompts |
| `src/utils/email_generator.py` | EmailGenerator class for generating and sending emails |
| `streamlit_app.py` (Email Campaign page) | UI for managing email campaigns |

## Tech Stack

- **Python 3.8+**
- **OpenAI/OpenRouter API** for LLM capabilities
- **Pandas** for data processing
- **scikit-learn** for ML models
- **pytest** for testing
- **Dataclasses** for type-safe schemas

## License

MIT
