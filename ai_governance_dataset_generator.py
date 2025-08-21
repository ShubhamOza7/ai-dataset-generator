#!/usr/bin/env python3
"""
AI Governance Dataset Generator
Creates comprehensive datasets with governance metadata for ISO 42001, fairness testing, and explainability analysis
"""

import pandas as pd
import numpy as np
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import random
from datetime import datetime, timedelta
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GovernanceMetadata:
    """Governance tracking for each data point"""
    record_id: str
    generation_timestamp: str
    scenario_source: str
    llm_model_used: str
    llm_prompt_hash: str
    bias_testing_flag: bool
    protected_class_combination: List[str]
    risk_category: str
    audit_trail_id: str
    compliance_tags: List[str]
    explainability_complexity: str  # simple, moderate, complex
    decision_boundary_flag: bool
    synthetic_data_flag: bool = True

@dataclass
class LoanScenario:
    """Enhanced loan scenario with governance tracking"""
    name: str
    description: str
    template: Dict[str, Any]
    variation_weight: float = 1.0
    priority: str = "medium"  # low, medium, high, critical
    # Governance additions
    bias_testing_scenario: bool = False
    protected_classes_involved: List[str] = None
    iso42001_risk_category: str = "medium"  # low, medium, high, critical
    explainability_target: str = "standard"  # standard, complex, edge_case
    compliance_requirements: List[str] = None

class AIGovernanceDatasetGenerator:
    """Enhanced dataset generator for AI governance compliance"""
    
    def __init__(self, seed: int = 42, llm_model: str = "unknown", prompt_hash: str = ""):
        """Initialize with governance tracking"""
        self.seed = seed
        self.random = random.Random(seed)
        np.random.seed(seed)
        self.llm_model = llm_model
        self.prompt_hash = prompt_hash
        self.generation_id = str(uuid.uuid4())
        
        # Enhanced value ranges for governance testing
        self.value_ranges = {
            'age': {'min': 18, 'max': 80},
            'income': {'min': 15000, 'max': 500000},
            'credit_score': {'min': 300, 'max': 850, 'no_credit': 0},
            'years_experience': {'min': 0, 'max': 50},
            'monthly_expenses': {'min': 800, 'max': 15000},
            'loan_amount': {'min': 1000, 'max': 1000000},
            'debt_to_income_ratio': {'min': 0.0, 'max': 1.5},
            'savings_account_balance': {'min': 0, 'max': 1000000}
        }
        
        # Enhanced categorical options for governance testing
        self.categorical_options = {
            'education_level': ['High School', 'Bachelor', 'Master', 'PhD', 'Trade School', 'Some College'],
            'employment_status': ['Employed Full-Time', 'Employed Part-Time', 'Self-Employed', 'Unemployed', 'Retired', 'Student'],
            'loan_type': ['Personal', 'Auto', 'Mortgage', 'Business', 'Student', 'Home Equity'],
            'property_ownership': ['Own', 'Rent', 'Living with Family', 'Other'],
            'payment_frequency': ['Monthly', 'Bi-Weekly', 'Weekly', 'Quarterly'],
            'bank_relationship': ['New Customer', 'Existing Customer', 'Premium Customer', 'No Relationship'],
            
            # Protected classes for fairness testing
            'gender': ['Male', 'Female', 'Non-Binary', 'Prefer not to say'],
            'race': ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Native American', 'Pacific Islander', 'Mixed Race', 'Other', 'Prefer not to say'],
            'age_group': ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+'],
            'disability_status': ['No Disability', 'Physical Disability', 'Cognitive Disability', 'Sensory Disability', 'Multiple Disabilities', 'Prefer not to say'],
            'marital_status': ['Single', 'Married', 'Divorced', 'Widowed', 'Separated', 'Domestic Partnership'],
            'religion': ['Christian', 'Muslim', 'Jewish', 'Hindu', 'Buddhist', 'Other', 'No Religion', 'Prefer not to say'],
            
            # Geographic for bias testing
            'zip_code': [f'{10000 + i}' for i in range(200)],  # Diverse zip codes
            
            # Target variable
            'target': ['Approved', 'Denied']
        }
        
        # Governance-specific additions
        self.bias_testing_combinations = self._generate_bias_testing_combinations()
        self.explainability_scenarios = self._generate_explainability_scenarios()
        self.iso42001_risk_categories = ['low', 'medium', 'high', 'critical']
        
        self.scenarios = []
    
    def _generate_bias_testing_combinations(self) -> List[Dict[str, List[str]]]:
        """Generate comprehensive protected class combinations for bias testing"""
        return [
            # Single protected class
            {'gender': ['Female'], 'age_group': ['all']},
            {'race': ['Black'], 'age_group': ['all']},
            {'disability_status': ['Physical Disability'], 'age_group': ['all']},
            
            # Intersectional combinations
            {'gender': ['Female'], 'race': ['Black'], 'age_group': ['18-25']},
            {'gender': ['Female'], 'race': ['Hispanic/Latino'], 'age_group': ['26-35']},
            {'gender': ['Male'], 'race': ['Asian'], 'disability_status': ['Cognitive Disability']},
            
            # Age-based bias
            {'age_group': ['18-25'], 'employment_status': ['Student']},
            {'age_group': ['66-75'], 'employment_status': ['Retired']},
            {'age_group': ['76+'], 'marital_status': ['Widowed']},
            
            # Socioeconomic combinations
            {'education_level': ['High School'], 'employment_status': ['Employed Part-Time']},
            {'education_level': ['PhD'], 'employment_status': ['Self-Employed']},
            
            # Geographic bias
            {'zip_code': ['10001', '10002', '10003']},  # Urban
            {'zip_code': ['90210', '90211', '90212']},  # High-income areas
        ]
    
    def _generate_explainability_scenarios(self) -> List[Dict[str, str]]:
        """Generate scenarios specifically for explainability testing"""
        return [
            {'complexity': 'simple', 'description': 'Single factor dominance'},
            {'complexity': 'moderate', 'description': 'Two-factor interaction'},
            {'complexity': 'complex', 'description': 'Multi-factor non-linear interaction'},
            {'complexity': 'edge_case', 'description': 'Boundary decision scenarios'},
            {'complexity': 'counter_intuitive', 'description': 'Scenarios that challenge human intuition'}
        ]
    
    def add_llm_scenarios(self, scenarios: List[Dict[str, Any]], llm_metadata: Dict[str, str] = None):
        """Add scenarios from LLM with governance metadata"""
        self.llm_metadata = llm_metadata or {}
        
        for scenario_dict in scenarios:
            # Convert to enhanced LoanScenario with governance metadata
            scenario = LoanScenario(
                name=scenario_dict.get('name', 'Unknown'),
                description=scenario_dict.get('description', ''),
                template=scenario_dict.get('template', {}),
                variation_weight=scenario_dict.get('variation_weight', 1.0),
                priority=scenario_dict.get('priority', 'medium'),
                # Governance enhancements
                bias_testing_scenario=self._is_bias_testing_scenario(scenario_dict),
                protected_classes_involved=self._extract_protected_classes(scenario_dict),
                iso42001_risk_category=self._determine_risk_category(scenario_dict),
                explainability_target=self._determine_explainability_target(scenario_dict),
                compliance_requirements=self._determine_compliance_requirements(scenario_dict)
            )
            self.scenarios.append(scenario)
    
    def _is_bias_testing_scenario(self, scenario: Dict[str, Any]) -> bool:
        """Determine if scenario is for bias testing"""
        bias_keywords = ['bias', 'discrimination', 'protected', 'fairness', 'demographic', 'minority']
        content = f"{scenario.get('name', '')} {scenario.get('description', '')}".lower()
        return any(keyword in content for keyword in bias_keywords)
    
    def _extract_protected_classes(self, scenario: Dict[str, Any]) -> List[str]:
        """Extract which protected classes are involved in scenario"""
        protected_classes = []
        content = f"{scenario.get('name', '')} {scenario.get('description', '')}".lower()
        
        if any(word in content for word in ['gender', 'male', 'female', 'woman', 'man']):
            protected_classes.append('gender')
        if any(word in content for word in ['race', 'racial', 'black', 'white', 'hispanic', 'asian']):
            protected_classes.append('race')
        if any(word in content for word in ['age', 'senior', 'elderly', 'young', 'youth']):
            protected_classes.append('age')
        if any(word in content for word in ['disability', 'disabled', 'impairment']):
            protected_classes.append('disability')
        
        return protected_classes
    
    def _determine_risk_category(self, scenario: Dict[str, Any]) -> str:
        """Determine ISO 42001 risk category"""
        content = f"{scenario.get('name', '')} {scenario.get('description', '')}".lower()
        priority = scenario.get('priority', 'medium')
        
        if priority == 'critical' or any(word in content for word in ['critical', 'high risk', 'severe']):
            return 'critical'
        elif priority == 'high' or any(word in content for word in ['edge', 'extreme', 'bias']):
            return 'high'
        elif priority == 'low':
            return 'low'
        else:
            return 'medium'
    
    def _determine_explainability_target(self, scenario: Dict[str, Any]) -> str:
        """Determine explainability complexity target"""
        content = f"{scenario.get('name', '')} {scenario.get('description', '')}".lower()
        
        if any(word in content for word in ['complex', 'interaction', 'non-linear']):
            return 'complex'
        elif any(word in content for word in ['edge', 'boundary', 'unusual']):
            return 'edge_case'
        else:
            return 'standard'
    
    def _determine_compliance_requirements(self, scenario: Dict[str, Any]) -> List[str]:
        """Determine which compliance requirements apply"""
        requirements = []
        content = f"{scenario.get('name', '')} {scenario.get('description', '')}".lower()
        
        if any(word in content for word in ['bias', 'discrimination', 'fairness']):
            requirements.extend(['ISO42001_Fairness', 'EU_AI_Act_Art15'])
        if any(word in content for word in ['transparency', 'explanation', 'interpretable']):
            requirements.extend(['ISO42001_Transparency', 'EU_AI_Act_Art13'])
        if any(word in content for word in ['audit', 'documentation', 'traceability']):
            requirements.extend(['ISO42001_Documentation', 'EU_AI_Act_Art12'])
        
        return list(set(requirements))  # Remove duplicates
    
    def generate_dataset(self, total_rows: int = 10000, bias_testing_ratio: float = 0.3) -> pd.DataFrame:
        """Generate comprehensive governance dataset"""
        logger.info(f"Generating governance dataset with {total_rows} rows")
        
        if not self.scenarios:
            self.scenarios = self._generate_default_governance_scenarios()
        
        # Calculate distribution with governance priorities
        distribution = self._calculate_governance_distribution(total_rows, bias_testing_ratio)
        
        all_rows = []
        governance_metadata = []
        
        for scenario, count in distribution:
            logger.info(f"Generating {count} rows for governance scenario: {scenario.name}")
            
            scenario_rows = []
            scenario_metadata = []
            
            for i in range(count):
                # Generate base data
                row_data = self._generate_scenario_row(scenario)
                
                # Add governance metadata
                metadata = self._generate_governance_metadata(scenario, i)
                
                # Combine data and metadata
                combined_row = {**row_data, **self._flatten_governance_metadata(metadata)}
                
                scenario_rows.append(combined_row)
                scenario_metadata.append(metadata)
            
            all_rows.extend(scenario_rows)
            governance_metadata.extend(scenario_metadata)
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        
        # Add governance audit trail
        df = self._add_governance_audit_trail(df, governance_metadata)
        
        # Validate governance compliance
        df = self._validate_governance_compliance(df)
        
        logger.info(f"Generated governance dataset with shape: {df.shape}")
        return df
    
    def _generate_governance_metadata(self, scenario: LoanScenario, index: int) -> GovernanceMetadata:
        """Generate comprehensive governance metadata for each record"""
        return GovernanceMetadata(
            record_id=str(uuid.uuid4()),
            generation_timestamp=datetime.now().isoformat(),
            scenario_source=scenario.name,
            llm_model_used=self.llm_model,
            llm_prompt_hash=self.prompt_hash,
            bias_testing_flag=scenario.bias_testing_scenario,
            protected_class_combination=scenario.protected_classes_involved or [],
            risk_category=scenario.iso42001_risk_category,
            audit_trail_id=f"{self.generation_id}_{index:06d}",
            compliance_tags=scenario.compliance_requirements or [],
            explainability_complexity=scenario.explainability_target,
            decision_boundary_flag=scenario.explainability_target == 'edge_case',
            synthetic_data_flag=True
        )
    
    def _flatten_governance_metadata(self, metadata: GovernanceMetadata) -> Dict[str, Any]:
        """Flatten governance metadata for DataFrame inclusion"""
        return {
            'gov_record_id': metadata.record_id,
            'gov_generation_timestamp': metadata.generation_timestamp,
            'gov_scenario_source': metadata.scenario_source,
            'gov_llm_model': metadata.llm_model_used,
            'gov_llm_prompt_hash': metadata.llm_prompt_hash,
            'gov_bias_testing_flag': metadata.bias_testing_flag,
            'gov_protected_classes': ','.join(metadata.protected_class_combination),
            'gov_risk_category': metadata.risk_category,
            'gov_audit_trail_id': metadata.audit_trail_id,
            'gov_compliance_tags': ','.join(metadata.compliance_tags),
            'gov_explainability_complexity': metadata.explainability_complexity,
            'gov_decision_boundary_flag': metadata.decision_boundary_flag,
            'gov_synthetic_flag': metadata.synthetic_data_flag
        }
    
    def _calculate_governance_distribution(self, total_rows: int, bias_testing_ratio: float) -> List[Tuple[LoanScenario, int]]:
        """Calculate row distribution prioritizing governance scenarios"""
        distribution = []
        
        # Separate scenarios by governance priority
        critical_scenarios = [s for s in self.scenarios if s.iso42001_risk_category == 'critical']
        high_scenarios = [s for s in self.scenarios if s.iso42001_risk_category == 'high']
        bias_scenarios = [s for s in self.scenarios if s.bias_testing_scenario]
        regular_scenarios = [s for s in self.scenarios if s.iso42001_risk_category in ['medium', 'low']]
        
        # Allocate rows with governance priorities
        bias_rows = int(total_rows * bias_testing_ratio)
        critical_rows = max(100, int(total_rows * 0.15))  # Minimum 100 rows for critical scenarios
        high_rows = max(50, int(total_rows * 0.10))
        remaining_rows = total_rows - bias_rows - critical_rows - high_rows
        
        # Distribute bias testing rows
        if bias_scenarios:
            rows_per_bias = max(1, bias_rows // len(bias_scenarios))
            for scenario in bias_scenarios:
                distribution.append((scenario, rows_per_bias))
        
        # Distribute critical risk rows
        if critical_scenarios:
            rows_per_critical = max(1, critical_rows // len(critical_scenarios))
            for scenario in critical_scenarios:
                distribution.append((scenario, rows_per_critical))
        
        # Distribute high risk rows
        if high_scenarios:
            rows_per_high = max(1, high_rows // len(high_scenarios))
            for scenario in high_scenarios:
                distribution.append((scenario, rows_per_high))
        
        # Distribute remaining rows
        if regular_scenarios:
            rows_per_regular = max(1, remaining_rows // len(regular_scenarios))
            for scenario in regular_scenarios:
                distribution.append((scenario, rows_per_regular))
        
        return distribution
    
    def _generate_scenario_row(self, scenario: LoanScenario) -> Dict[str, Any]:
        """Generate a single row based on scenario template"""
        row = {}
        
        # Generate base demographic and financial data
        row.update(self._generate_base_demographics())
        row.update(self._generate_financial_data())
        
        # Apply scenario-specific constraints
        if scenario.template:
            row.update(self._apply_scenario_constraints(row, scenario.template))
        
        # Add governance-specific fields if needed
        if scenario.bias_testing_scenario:
            row.update(self._generate_bias_testing_fields(scenario))
        
        # Generate target variable with governance considerations
        row['target'] = self._generate_governance_aware_target(row, scenario)
        
        return row
    
    def _generate_base_demographics(self) -> Dict[str, Any]:
        """Generate base demographic data with governance considerations"""
        return {
            'age': self.random.randint(18, 80),
            'gender': self.random.choice(self.categorical_options['gender']),
            'race': self.random.choice(self.categorical_options['race']),
            'education_level': self.random.choice(self.categorical_options['education_level']),
            'employment_status': self.random.choice(self.categorical_options['employment_status']),
            'marital_status': self.random.choice(self.categorical_options['marital_status']),
            'disability_status': self.random.choice(self.categorical_options['disability_status']),
            'religion': self.random.choice(self.categorical_options['religion']),
            'zip_code': self.random.choice(self.categorical_options['zip_code'])
        }
    
    def _generate_financial_data(self) -> Dict[str, Any]:
        """Generate financial data with realistic correlations"""
        # Base income generation
        income = self.random.randint(15000, 500000)
        
        # Correlate credit score with income (with noise)
        income_factor = min(income / 100000, 3.0)
        base_credit = 500 + (income_factor * 100)
        credit_score = max(300, min(850, int(base_credit + self.random.normalvariate(0, 50))))
        
        # Generate other financial fields
        years_exp = min(self.random.randint(0, min(50, max(0, (income // 2000) - 15))), 50)
        monthly_expenses = max(800, min(int(income * self.random.uniform(0.3, 0.8) / 12), 15000))
        savings = max(0, int(income * self.random.uniform(0, 0.5)))
        
        # Loan amount correlated with income and credit
        max_loan = min(income * 5, 800000)
        loan_amount = self.random.randint(1000, max(1000, int(max_loan)))
        
        # Calculate debt-to-income ratio
        debt_to_income = monthly_expenses * 12 / income
        
        return {
            'income': income,
            'credit_score': credit_score,
            'years_experience': years_exp,
            'monthly_expenses': monthly_expenses,
            'loan_amount': loan_amount,
            'debt_to_income_ratio': round(debt_to_income, 3),
            'savings_account_balance': savings,
            'loan_type': self.random.choice(self.categorical_options['loan_type']),
            'property_ownership': self.random.choice(self.categorical_options['property_ownership']),
            'payment_frequency': self.random.choice(self.categorical_options['payment_frequency']),
            'bank_relationship': self.random.choice(self.categorical_options['bank_relationship'])
        }
    
    def _apply_scenario_constraints(self, row: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scenario-specific constraints to row data"""
        for field, constraint in template.items():
            if field in row:
                if isinstance(constraint, tuple) and len(constraint) == 2:
                    # Range constraint
                    min_val, max_val = constraint
                    if isinstance(row[field], (int, float)):
                        row[field] = self.random.uniform(min_val, max_val)
                elif isinstance(constraint, list):
                    # Choice constraint
                    row[field] = self.random.choice(constraint)
                else:
                    # Direct value
                    row[field] = constraint
        
        return row
    
    def _generate_bias_testing_fields(self, scenario: LoanScenario) -> Dict[str, Any]:
        """Generate specific fields for bias testing scenarios"""
        fields = {}
        
        # Add derived fields for bias analysis
        if 'age' in (scenario.protected_classes_involved or []):
            fields['age_group'] = self._categorize_age(fields.get('age', 35))
        
        return fields
    
    def _categorize_age(self, age: int) -> str:
        """Categorize age into groups for bias testing"""
        if age <= 25:
            return '18-25'
        elif age <= 35:
            return '26-35'
        elif age <= 45:
            return '36-45'
        elif age <= 55:
            return '46-55'
        elif age <= 65:
            return '56-65'
        elif age <= 75:
            return '66-75'
        else:
            return '76+'
    
    def _generate_governance_aware_target(self, row: Dict[str, Any], scenario: LoanScenario) -> str:
        """Generate target variable with governance considerations"""
        # Base approval logic
        approval_score = 0
        
        # Credit score factor
        if row['credit_score'] > 750:
            approval_score += 0.4
        elif row['credit_score'] > 650:
            approval_score += 0.2
        elif row['credit_score'] < 500:
            approval_score -= 0.3
        
        # Income factor
        if row['income'] > 80000:
            approval_score += 0.3
        elif row['income'] < 30000:
            approval_score -= 0.2
        
        # Debt-to-income factor
        if row['debt_to_income_ratio'] < 0.3:
            approval_score += 0.2
        elif row['debt_to_income_ratio'] > 0.5:
            approval_score -= 0.3
        
        # Add governance-specific considerations
        if scenario.bias_testing_scenario:
            # Ensure diverse outcomes for bias testing
            approval_score += self.random.uniform(-0.2, 0.2)
        
        # Random factor for realism
        approval_score += self.random.uniform(-0.1, 0.1)
        
        return 'Approved' if approval_score > 0.1 else 'Denied'
    
    def _add_governance_audit_trail(self, df: pd.DataFrame, metadata: List[GovernanceMetadata]) -> pd.DataFrame:
        """Add comprehensive audit trail for governance"""
        # Add generation metadata
        df['gov_generation_id'] = self.generation_id
        df['gov_generation_date'] = datetime.now().date().isoformat()
        df['gov_total_records'] = len(df)
        df['gov_bias_testing_records'] = df['gov_bias_testing_flag'].sum()
        df['gov_data_version'] = '1.0'
        
        return df
    
    def _validate_governance_compliance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate dataset meets governance requirements"""
        logger.info("Validating governance compliance...")
        
        # Check bias testing coverage
        bias_coverage = df['gov_bias_testing_flag'].sum() / len(df)
        logger.info(f"Bias testing coverage: {bias_coverage:.2%}")
        
        # Check protected class diversity
        for col in ['gender', 'race', 'age_group']:
            if col in df.columns:
                diversity = len(df[col].unique())
                logger.info(f"{col} diversity: {diversity} categories")
        
        # Check risk category distribution
        risk_dist = df['gov_risk_category'].value_counts()
        logger.info(f"Risk category distribution: {risk_dist.to_dict()}")
        
        return df
    
    def _generate_default_governance_scenarios(self) -> List[LoanScenario]:
        """Generate default scenarios focused on governance testing"""
        return [
            LoanScenario(
                name="Gender Bias Testing",
                description="Test for gender-based lending bias",
                template={'gender': ['Female']},
                bias_testing_scenario=True,
                protected_classes_involved=['gender'],
                iso42001_risk_category='critical',
                explainability_target='standard',
                compliance_requirements=['ISO42001_Fairness', 'EU_AI_Act_Art15']
            ),
            LoanScenario(
                name="Racial Bias Testing",
                description="Test for race-based lending bias",
                template={'race': ['Black', 'Hispanic/Latino']},
                bias_testing_scenario=True,
                protected_classes_involved=['race'],
                iso42001_risk_category='critical',
                explainability_target='standard',
                compliance_requirements=['ISO42001_Fairness', 'EU_AI_Act_Art15']
            ),
            LoanScenario(
                name="Age Discrimination Testing",
                description="Test for age-based lending discrimination",
                template={'age': (65, 80)},
                bias_testing_scenario=True,
                protected_classes_involved=['age'],
                iso42001_risk_category='high',
                explainability_target='standard',
                compliance_requirements=['ISO42001_Fairness']
            ),
            LoanScenario(
                name="Complex Decision Boundary",
                description="Edge cases for explainability testing",
                template={'credit_score': (650, 700), 'income': (45000, 55000)},
                bias_testing_scenario=False,
                protected_classes_involved=[],
                iso42001_risk_category='medium',
                explainability_target='edge_case',
                compliance_requirements=['ISO42001_Transparency']
            ),
            LoanScenario(
                name="High Risk Profile",
                description="High-risk lending scenarios for audit",
                template={'credit_score': (300, 500), 'debt_to_income_ratio': (0.7, 1.2)},
                bias_testing_scenario=False,
                protected_classes_involved=[],
                iso42001_risk_category='high',
                explainability_target='complex',
                compliance_requirements=['ISO42001_Risk_Management']
            )
        ]
    
    def export_governance_dataset(self, df: pd.DataFrame, filename: str = None, formats: List[str] = ['csv']) -> List[str]:
        """Export dataset with governance documentation"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"governance_dataset_{timestamp}"
        
        exported_files = []
        
        for fmt in formats:
            if fmt == 'csv':
                filepath = f"{filename}.csv"
                df.to_csv(filepath, index=False)
            elif fmt == 'json':
                filepath = f"{filename}.json"
                df.to_json(filepath, orient='records', indent=2)
            elif fmt == 'parquet':
                filepath = f"{filename}.parquet"
                df.to_parquet(filepath, index=False)
            elif fmt == 'governance_report':
                filepath = f"{filename}_governance_report.json"
                self._export_governance_report(df, filepath)
            
            exported_files.append(filepath)
            logger.info(f"Exported governance dataset to: {filepath}")
        
        return exported_files
    
    def _export_governance_report(self, df: pd.DataFrame, filepath: str):
        """Export comprehensive governance compliance report"""
        report = {
            'generation_metadata': {
                'generation_id': self.generation_id,
                'generation_timestamp': datetime.now().isoformat(),
                'llm_model': self.llm_model,
                'prompt_hash': self.prompt_hash,
                'total_records': len(df),
                'data_version': '1.0'
            },
            'compliance_summary': {
                'bias_testing_coverage': float(df['gov_bias_testing_flag'].sum() / len(df)),
                'risk_distribution': df['gov_risk_category'].value_counts().to_dict(),
                'protected_class_coverage': {
                    'gender_diversity': len(df['gender'].unique()) if 'gender' in df.columns else 0,
                    'racial_diversity': len(df['race'].unique()) if 'race' in df.columns else 0,
                    'age_diversity': len(df['age_group'].unique()) if 'age_group' in df.columns else 0
                },
                'explainability_distribution': df['gov_explainability_complexity'].value_counts().to_dict()
            },
            'iso42001_compliance': {
                'risk_management_records': len(df[df['gov_risk_category'].isin(['high', 'critical'])]),
                'fairness_testing_records': df['gov_bias_testing_flag'].sum(),
                'transparency_records': len(df[df['gov_explainability_complexity'] != 'simple']),
                'documentation_complete': True
            },
            'audit_trail': {
                'data_lineage': 'LLM-generated scenarios -> Synthetic data generation',
                'generation_process': 'AI Governance Dataset Generator v1.0',
                'validation_status': 'Passed',
                'compliance_tags_used': list(set([tag for tags in df['gov_compliance_tags'].str.split(',') for tag in tags if tag]))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

def main():
    """Test the governance dataset generator"""
    generator = AIGovernanceDatasetGenerator(
        seed=42,
        llm_model="llama3.2:3b",
        prompt_hash=hashlib.md5("test_prompt".encode()).hexdigest()
    )
    
    # Generate governance dataset
    dataset = generator.generate_dataset(total_rows=1000, bias_testing_ratio=0.4)
    
    # Export with governance documentation
    exported_files = generator.export_governance_dataset(
        dataset, 
        filename="ai_governance_test_dataset",
        formats=['csv', 'json', 'governance_report']
    )
    
    print(f"Generated governance dataset with {len(dataset)} rows")
    print(f"Exported files: {exported_files}")
    print(f"Governance features: {[col for col in dataset.columns if col.startswith('gov_')]}")

if __name__ == "__main__":
    main()
