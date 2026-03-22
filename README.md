# Churn Risk Analysis Pipeline

An agentic AI system that detects customer churn risk in real time,
matches against historical patterns, and generates actionable retention plans.

## What This System Does

Reads a customer email or support ticket and automatically:
1. Detects warning signals in the message
2. Finds similar historical cases from 440,833 customer records
3. Generates a personalised retention action plan

## The Three Agents

| Agent | Name | Job |
|-------|------|-----|
| Agent 1 | Multi-Channel Analyser | Reads customer messages and spots risk signals |
| Agent 2 | Experience Curator | Finds similar past customers using vector search |
| Agent 3 | Playbook Creator | Generates retention action plans using an LLM |

## Dataset

440,833 customer records with features:
Age, Gender, Tenure, Usage Frequency, Support Calls,
Payment Delay, Subscription Type, Contract Length,
Total Spend, Last Interaction, Churn

## Validated Signal Thresholds

| Signal | Threshold | Detection Rate | Weight |
|--------|-----------|---------------|--------|
| Support Calls | >= 4 | 65.9% | 0.35 |
| Total Spend | <= 600 | 57.2% | 0.25 |
| Payment Delay | >= 17 days | 46.2% | 0.20 |
| Last Interaction | >= 15 days | 54.0% | 0.15 |
| Usage Frequency | < 14 days/month | weak | 0.05 |

## Build Progress

- [x] Phase 1 -- Data exploration and signal threshold validation
- [x] Phase 2 -- Simulated customer email generation
- [ ] Phase 3 -- Agent 1: Multi-Channel Analyser
- [ ] Phase 4 -- Agent 2: Experience Curator
- [ ] Phase 5 -- Agent 3: Playbook Creator
- [ ] Phase 6 -- Orchestration: wire all agents together
- [ ] Phase 7 -- Feedback loop: continuous learning
- [ ] Phase 8 -- Production: API, Docker, CI/CD

## Tech Stack

- Python 3.10+
- pandas -- data manipulation
- OpenRouter API -- LLM access (Claude 3 Haiku)
- ChromaDB -- vector database (Phase 4)
- FastAPI -- production API (Phase 8)

## Author

Built from scratch as a learning project in agentic AI engineering.
