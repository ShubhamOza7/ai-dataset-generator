# 🏛️ AI Governance Dataset Generator

A production-ready AI governance tool that generates comprehensive datasets with compliance metadata for **ISO 42001**, **fairness testing**, and **explainability analysis**.

## 🎯 What it does

**Foundation dataset generator for complete AI governance pipeline:**

- **🏛️ ISO 42001 Compliance**: Risk management, audit trails, governance documentation
- **⚖️ Fairness Testing**: Protected class combinations, bias detection scenarios
- **🔬 Explainability**: Decision boundary cases, complex interaction patterns
- **📋 Complete Audit Trail**: LLM interactions, governance metadata, compliance tracking
- **🎯 Governance-First Design**: Every record tagged with compliance requirements

**Perfect foundation for Steps 2, 3, 4:**

- **Step 2**: ISO 42001 audit framework (uses governance metadata)
- **Step 3**: Fairness testing suite (uses bias testing records)
- **Step 4**: Explainability analyzer (uses complexity classifications)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install LLM (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models for governance analysis
ollama pull llama3.2:3b      # Fast, good governance coverage
ollama pull gpt-oss:20b      # Comprehensive governance scenarios
```

### 3. Generate Governance Dataset

**Simple Governance Dataset:**

```bash
python3 auto_dataset_cli.py age income credit_score employment_status target \
  --context "loan approval" \
  --size 1000 \
  --formats csv governance_report
```

**Full Governance Dataset:**

```bash
python3 auto_dataset_cli.py \
  age income credit_score years_experience monthly_expenses loan_amount \
  debt_to_income_ratio savings_account_balance education_level \
  employment_status loan_type property_ownership payment_frequency \
  bank_relationship gender race age_group disability_status \
  marital_status religion zip_code target \
  --context "loan approval" \
  --size 5000 \
  --formats csv json governance_report \
  --llm-model "llama3.2:3b"
```

### 4. Quick Testing (No LLM)

```bash
python3 quick_dataset_cli.py age income credit_score target \
  --context "loan approval" \
  --size 100
```

## 🏛️ AI Governance Features

### ISO 42001 Compliance Ready

- **Risk categorization**: low/medium/high/critical for each record
- **Audit trail IDs**: Complete traceability from LLM prompt to data point
- **Governance metadata**: Embedded in every record
- **Documentation**: Automatic compliance report generation

### Fairness Testing Optimized

- **Protected class combinations**: Gender, race, age, disability intersections
- **Bias testing flags**: 40% of records specifically for bias detection
- **Demographic parity scenarios**: Systematic coverage of all groups
- **Historical discrimination patterns**: Known bias scenarios for testing

### Explainability Enhanced

- **Complexity classification**: Simple/moderate/complex/edge_case per record
- **Decision boundary flags**: Edge cases requiring explanation
- **Feature interaction patterns**: Multi-factor decision scenarios
- **Stakeholder explanation targets**: Different explanation needs

## 📊 Governance Dataset Structure

### Core Features

```
Standard features: age, income, credit_score, target, etc.
```

### Governance Metadata (Added to Every Record)

```
gov_record_id              # Unique identifier
gov_generation_timestamp   # When record was created
gov_scenario_source        # Which LLM scenario generated this
gov_llm_model              # Which LLM model was used
gov_llm_prompt_hash        # Hash of the prompt used
gov_bias_testing_flag      # True if for bias testing
gov_protected_classes      # Which protected classes involved
gov_risk_category          # ISO 42001 risk level
gov_audit_trail_id        # Complete audit trail
gov_compliance_tags        # Applicable compliance requirements
gov_explainability_complexity  # Explanation difficulty
gov_decision_boundary_flag # True if edge case
gov_synthetic_flag         # Data provenance
```

### Compliance Report (Auto-Generated)

```json
{
  "iso42001_compliance": {
    "risk_management": {...},
    "fairness_testing": {...},
    "transparency": {...}
  },
  "eu_ai_act_compliance": {
    "article_13_transparency": {...},
    "article_15_fairness": {...}
  },
  "recommendations": {...}
}
```

## 🎯 Example Governance Output

**Generated Governance Scenarios:**

- Gender Bias Testing - Female Applicants (Critical Risk)
- Racial Bias Testing - Minority Groups (Critical Risk)
- Age Discrimination - Senior Citizens (High Risk)
- Complex Decision Boundary Analysis (Edge Case)
- High-Risk Decision Documentation (Audit Trail)
- Intersectional Bias Testing (Multiple Protected Classes)

**Compliance Metrics:**

```
📊 Dataset Metrics:
• Total Records: 5,000
• Bias Testing Records: 2,000 (40%)
• Critical Risk Records: 750 (15%)

🎯 Compliance Coverage:
• ISO 42001: ✅ Ready
• Bias Testing: ✅ Ready
• Explainability: ✅ Ready
• Audit Trail: ✅ Complete
```

**File Outputs:**

- `governance_loan_approval_20250821_120500.csv` - Main dataset with governance metadata
- `governance_loan_approval_20250821_120500_compliance_report.json` - Full compliance documentation
- `governance_llm_interactions/governance_interaction_20250821_120500.txt` - Complete LLM audit trail

## 🔗 AI Governance Pipeline Integration

This tool is **Step 1** of a complete AI governance ecosystem:

### **Step 1: ai-dataset-generator** ✅ (This Repository)

- Generates governance-ready datasets
- Embeds compliance metadata
- Creates audit trails

### **Step 2: iso42001-audit-framework** 🔄 (Next)

- Reads governance metadata
- Automated compliance checking
- Risk assessment validation

### **Step 3: fairness-testing-suite** 🔄 (Next)

- Uses bias_testing_flag records
- Protected class analysis
- Demographic parity testing

### **Step 4: explainability-analyzer** 🔄 (Next)

- Uses explainability_complexity fields
- Decision boundary analysis
- Stakeholder explanations

## 📁 Project Structure

### Core Files

- `auto_dataset_cli.py` - Main governance dataset CLI
- `ai_governance_dataset_generator.py` - Enhanced generator with governance metadata
- `quick_dataset_cli.py` - Quick testing without LLM
- `requirements.txt` - Dependencies

### Generated Outputs

- `governance_datasets/` - Datasets with embedded governance metadata
- `governance_llm_interactions/` - Complete LLM audit trails for compliance

## 🛠️ Advanced Governance Usage

### Custom Compliance Requirements

```bash
# Focus on bias testing
--size 2000 # Smaller, bias-focused dataset

# Complex explainability scenarios
--llm-model "gpt-oss:20b" # More sophisticated scenarios
```

### Integration with Next Steps

```python
# Load governance dataset for Step 2 (ISO 42001 audit)
df = pd.read_csv('governance_loan_approval.csv')
critical_records = df[df['gov_risk_category'] == 'critical']

# For Step 3 (fairness testing)
bias_test_records = df[df['gov_bias_testing_flag'] == True]

# For Step 4 (explainability)
complex_cases = df[df['gov_explainability_complexity'] == 'edge_case']
```

## 🎯 Production Governance Benefits

### Regulatory Compliance

- **EU AI Act Article 13**: Transparency requirements covered
- **EU AI Act Article 15**: Fairness testing data ready
- **ISO 42001**: Complete management system data

### Risk Management

- **Risk-categorized records**: Focus testing on high-risk scenarios
- **Complete audit trails**: From LLM prompt to final decision
- **Documentation automation**: Compliance reports auto-generated

### Quality Assurance

- **Systematic bias coverage**: All protected class combinations
- **Explainability readiness**: Pre-classified complexity levels
- **Data provenance**: Complete generation lineage

Perfect foundation for building **production-ready AI governance systems** that ensure compliance, fairness, and explainability! 🏛️⚖️🔬
