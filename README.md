# AI Dataset Generator

**Generic AI Dataset Generator on AWS with Reusable LLM Core Services**

Architected a generic AI Dataset Generator on AWS using reusable LLM core services, allowing users to execute dynamic model selection and integration to synthesize structured datasets directly from user-provided context prompts.

## Overview

This project provides a comprehensive framework for generating high-quality, structured datasets for AI/ML applications. Built on AWS infrastructure with a focus on scalability, compliance, and data integrity.

### Key Features

- **Dynamic Model Selection**: Seamlessly integrate and switch between multiple LLM providers (Llama, GPT-OSS, custom models)
- **Reusable LLM Core Services**: Modular architecture enabling consistent LLM interactions across different use cases
- **Context-Driven Synthesis**: Generate structured datasets directly from user-provided context prompts
- **Multiple Output Formats**: CSV, JSON, Parquet for different downstream analysis tools

## MLOps Pipeline

Engineered robust MLOps pipelines in Python featuring:

### RAG (Retrieval-Augmented Generation)
- Context-aware scenario discovery using LLM-powered analysis
- Domain-specific knowledge retrieval for realistic data patterns
- Enhanced prompt engineering for comprehensive coverage

### Fine-Tuning Capabilities
- Customizable model parameters for domain-specific optimization
- Support for multiple LLM backends with configurable settings
- Iterative improvement through feedback loops

### Guardrails
- Input validation and sanitization for user prompts
- Output quality checks ensuring data integrity
- Compliance-ready audit trails for all LLM interactions

### Automated Evaluation
- Data quality validation strategies
- Consistency checks for generated datasets
- Comprehensive logging for transparency and debugging

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install LLM (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2:3b      # Fast, good coverage
ollama pull gpt-oss:20b      # More comprehensive scenarios
```

### 3. Generate Test Dataset

**Simple Usage:**

```bash
python3 auto_dataset_cli.py age income credit_score employment_status target \
  --context "loan approval" \
  --size 1000 \
  --formats csv
```

**Full Feature Set:**

```bash
python3 auto_dataset_cli.py \
  age income credit_score years_experience monthly_expenses loan_amount \
  debt_to_income_ratio savings_account_balance education_level \
  employment_status loan_type property_ownership payment_frequency \
  bank_relationship gender race age_group disability_status \
  marital_status religion zip_code target \
  --context "loan approval" \
  --size 5000 \
  --formats csv json \
  --llm-model "llama3.2:3b"
```

### 4. Quick Testing (No LLM)

```bash
python3 quick_dataset_cli.py age income credit_score target \
  --context "loan approval" \
  --size 100
```

## Generated Dataset Features

### Core Data
- All your requested features (age, income, credit_score, target, etc.)
- Realistic relationships between features
- Mathematically consistent derived fields

### Comprehensive Test Coverage
- **Bias Testing Data**: All demographic combinations (gender x race x age, etc.)
- **Edge Cases**: Boundary values, unusual but valid combinations
- **Stress Testing**: Extreme values within realistic ranges
- **Domain Scenarios**: High-risk, low-risk, typical approval patterns

### Data Quality
- **Realistic Correlations**: Income correlates with loan amounts, age with experience
- **Business Logic**: Debt-to-income ratios calculated correctly
- **Diverse Scenarios**: 20+ different testing scenarios from LLM analysis

## Example Generated Scenarios

LLM discovers scenarios like:

- Young Professional (age 22-30, starting income, student loans)
- Senior Citizen (age 65+, retirement income, established credit)
- High-Risk Applicant (poor credit, high debt-to-income)
- Protected Class Testing (female + minority combinations)
- Geographic Bias Testing (different zip codes)
- Edge Cases (young person with high income, senior with no credit history)

Each scenario generates realistic test data that downstream tools can analyze.

## Project Structure

### Core Files
- `auto_dataset_cli.py` - Main CLI for AI-powered dataset generation
- `ai_governance_dataset_generator.py` - Core generation engine
- `quick_dataset_cli.py` - Quick testing without LLM
- `requirements.txt` - Dependencies

### Examples
- `examples/governance_prompts/` - Sample LLM interactions and outputs

## AI Governance Pipeline Integration

This is **Step 1** of the AI governance pipeline:

### Step 1: ai-dataset-generator (This Repository)
- Generates comprehensive test datasets
- All demographic combinations for bias testing
- Edge cases for stress testing
- Realistic scenarios for validation

### Step 2: iso42001-audit-framework (Next)
- Takes the generated dataset
- Performs ISO 42001 compliance auditing
- Risk assessment and documentation

### Step 3: fairness-testing-suite (Next)
- Takes the generated dataset
- Tests for bias across demographics
- Fairness metrics and reporting

### Step 4: explainability-analyzer (Next)
- Takes the generated dataset
- Explains model decisions
- Stakeholder-friendly interpretations

## Advanced Usage

### Different Domains

```bash
# Fraud detection
python3 auto_dataset_cli.py transaction_amount user_age account_type target \
  --context "fraud detection" --size 2000

# Hiring decisions
python3 auto_dataset_cli.py education experience skills interview_score target \
  --context "hiring decisions" --size 1500
```

### Custom Models

```bash
# Use different LLM for scenario discovery
--llm-model "gpt-oss:20b"    # More comprehensive
--llm-model "llama3.1"       # Alternative model
```

### Multiple Output Formats

```bash
--formats csv json parquet   # For different downstream tools
```

## Architecture Highlights

### AWS Integration
- Designed for deployment on AWS infrastructure
- Scalable architecture supporting high-throughput dataset generation
- Cloud-native design patterns for reliability and performance

### Reusable LLM Core Services
- Abstracted LLM interaction layer for vendor-agnostic operations
- Configurable model endpoints and parameters
- Centralized prompt management and optimization

### Data Quality Assurance
- Automated validation pipelines for generated data
- Integrity checks ensuring compliance-ready outputs
- Comprehensive audit trails for all generation processes

## Why This Approach Works

### Comprehensive Coverage
- LLM discovers scenarios humans might miss
- Systematic coverage of all demographic combinations
- Domain expertise applied to realistic scenario generation

### Production Ready
- Realistic data relationships (not just random values)
- Business logic embedded (loan amounts match income levels)
- Mathematical consistency (calculated fields are correct)

### Flexible Foundation
- Works for any domain (loan approval, fraud detection, hiring, etc.)
- Generates data patterns needed for any downstream analysis
- Modular design - each pipeline step is independent

## Next Steps

1. **Generate your test dataset** with this tool
2. **Build ISO 42001 audit framework** to analyze the data
3. **Create fairness testing suite** to test bias in the data
4. **Develop explainability analyzer** to explain decisions on the data

Each step builds on the comprehensive test data generated here.
