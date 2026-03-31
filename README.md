# Customer Churn Prediction Pipeline

An end-to-end customer churn analysis pipeline using **Agentic AI** - featuring specialized AI agents that collaborate to detect churn signals, assess risk, and recommend retention actions.

## Architecture

The pipeline uses a multi-agent architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                       │
│              (Coordinates the entire pipeline)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────────┐ ┌──────────┐ ┌──────────┐
│ Signal Agent  │ │ Risk     │ │ Action   │
│               │ │ Agent    │ │ Agent    │
│ Detects churn│ │ Assesses │ │ Generates│
│ signals from │ │ risk     │ │ retention│
│ customer data│ │ level    │ │ actions  │
└───────────────┘ └──────────┘ └──────────┘
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

# Show data summary
python main.py summary
```

## Features

- **Signal Detection**: Identifies behavioral patterns indicating churn risk
- **Risk Assessment**: Calculates risk scores and probability
- **Retention Actions**: Generates personalized recommendations
- **LLM Enhancement**: Uses AI for deeper semantic analysis

## Project Structure

```
churn-pipeline/
├── src/
│   ├── agents/          # AI agents (signal, risk, action, orchestrator)
│   ├── schemas/         # Data models and types
│   ├── utils/          # Utilities (data loading, output formatting)
│   └── config.py       # Configuration and thresholds
├── data/
│   ├── raw/            # Raw customer data
│   ├── processed/      # Cleaned data
│   └── simulated/      # Generated content
├── outputs/            # Analysis results
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

### Single customer analysis
```bash
python main.py single --customer '{"Age": 35, "Tenure": 12, "Support Calls": 5, ...}'
```

## Tech Stack

- **Python 3.8+**
- **OpenAI/OpenRouter API** for LLM capabilities
- **Pandas** for data processing
- **Dataclasses** for type-safe schemas
