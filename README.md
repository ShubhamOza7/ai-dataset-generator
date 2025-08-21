# 🚀 Automated Dataset Generation System

A production-ready tool that automatically generates comprehensive testing datasets from feature names using AI-powered scenario discovery.

## What it does

Takes your feature names and domain context → Generates production-quality testing datasets automatically:

- **🧠 AI Scenario Discovery**: Uses LLMs to discover all possible testing scenarios
- **📊 Comprehensive Coverage**: Generates bias testing, edge cases, boundary conditions
- **🎯 Domain-Specific**: Tailored datasets for loan approval, fraud detection, etc.
- **⚡ Multiple Formats**: Outputs CSV, JSON, Parquet, Excel
- **🔍 Complete Audit Trail**: Logs all LLM interactions for debugging/improvement

Perfect for ML engineers who need comprehensive test datasets without manual scenario design!

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install LLM (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (choose one)
ollama pull llama3.2:3b      # Fast, good for testing
ollama pull gpt-oss:20b      # Larger, more comprehensive
```

### 3. Generate Dataset

**Simple Usage:**

```bash
python3 auto_dataset_cli.py age income credit_score employment_status loan_type target --context "loan approval" --size 1000 --formats csv json
```

**Full Example:**

```bash
python3 auto_dataset_cli.py \
  age income credit_score years_experience monthly_expenses loan_amount \
  debt_to_income_ratio savings_account_balance education_level \
  employment_status loan_type property_ownership payment_frequency \
  bank_relationship gender race age_group disability_status \
  marital_status religion zip_code target \
  --context "loan approval" \
  --size 1000 \
  --formats csv json \
  --llm-model "llama3.2:3b"
```

### 4. Quick Testing (No LLM)

```bash
python3 quick_dataset_cli.py age income credit_score target --context "loan approval" --size 100
```

## 🎯 How It Works

### Step 1: AI Scenario Discovery

The system sends your features to an LLM with a comprehensive prompt that discovers:

- **Demographic combinations** for bias testing
- **Domain-specific profiles** (high/low risk, different income levels)
- **Edge cases** (unusual but valid combinations)
- **Stress testing** (boundary values, extreme cases)

### Step 2: Intelligent Dataset Generation

For each discovered scenario:

- **Realistic data generation** with proper correlations
- **Mathematical consistency** (debt-to-income ratios, age vs experience)
- **Domain expertise** (loan amounts match income levels)
- **Bias testing coverage** (protected classes, geographic variations)

### Step 3: Production Export

- Multiple file formats
- Comprehensive statistics
- Full LLM interaction logs
- Ready for immediate ML testing

## 🔧 Advanced Usage

### Custom LLM Models

```bash
# Use different models
--llm-model "llama3.1"
--llm-model "mistral"
--llm-model "gpt-oss:20b"
```

### Multiple Output Formats

```bash
--formats csv json parquet excel
```

### Custom Output Directory

```bash
--output-dir "my_datasets"
```

## 📁 Project Structure

### Core Files

- `auto_dataset_cli.py` - Main automated CLI with LLM integration
- `loan_dataset_generator.py` - Core dataset generation logic
- `quick_dataset_cli.py` - Quick testing without LLM (uses predefined scenarios)
- `requirements.txt` - Python dependencies

### Generated Files

- `generated_datasets/` - Output datasets in multiple formats
- `llm_interactions/` - Complete LLM prompt/response logs for debugging

### Test Files

- `model_v2_final.pkl` - Sample model for testing extraction

## 🎯 Example Output

**Generated Scenarios (20+ discovered by LLM):**

- Young Professional
- Senior Citizen Bias Testing
- Historical Discrimination Patterns
- High-Value Homebuyer
- Subprime Borrower
- Geographic Bias Testing
- Extreme Credit Score Edge Cases
- Mathematical Boundary Conditions

**Dataset Statistics:**

```
📊 Generated Dataset:
• Rows: 1,000
• Columns: 22
• Scenarios: 20
• Bias testing coverage: Comprehensive

🎯 Target Distribution:
• Approved: 486
• Denied: 514
```

**File Outputs:**

- `loan_approval_20250821_120500.csv` - Main dataset
- `loan_approval_20250821_120500.json` - JSON format
- `llm_interactions/interaction_20250821_120500.txt` - Full LLM log

## 🚀 Production Features

### Comprehensive Testing Coverage

- **Bias Testing**: Age, gender, race, disability status combinations
- **Edge Cases**: Unusual but valid data combinations
- **Stress Testing**: Boundary values and extreme scenarios
- **Domain Expertise**: Realistic financial relationships

### Quality Assurance

- **Mathematical Consistency**: All derived fields calculated correctly
- **Business Logic**: Loan amounts correlate with income and credit
- **Constraint Validation**: All values within realistic ranges
- **Audit Trail**: Complete LLM interaction logs

### Scalability

- **Multiple LLM Models**: Choose speed vs. comprehensiveness
- **Flexible Output**: 100 rows for testing to 100K+ for production
- **Multiple Formats**: Direct integration with any ML pipeline

## 🛠️ Troubleshooting

### LLM Issues

```bash
# Check Ollama is running
ollama list

# Test model
ollama run llama3.2:3b "Hello"
```

### Performance Tips

- Use `llama3.2:3b` for speed
- Use `gpt-oss:20b` for comprehensive scenarios
- Start with small datasets (--size 100) for testing

### Common Issues

- **Timeout errors**: Use smaller models or increase timeout
- **Memory issues**: Reduce dataset size or use quick_dataset_cli.py
- **Missing features**: Check that all requested features are supported

Perfect for ML engineers who need production-ready test datasets without manual scenario design! 🎯📊
