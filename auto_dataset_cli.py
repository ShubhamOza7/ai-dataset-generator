#!/usr/bin/env python3
"""
AI Governance Dataset Generation CLI
Takes feature names → Generates comprehensive governance datasets for ISO 42001, fairness testing, and explainability
"""

import argparse
import json
import subprocess
import sys
import os
import tempfile
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Import our enhanced governance generator
from ai_governance_dataset_generator import AIGovernanceDatasetGenerator, LoanScenario

console = Console()

class AIGovernanceDatasetCLI:
    """AI Governance Dataset Generation CLI"""
    
    def __init__(self, llm_model: str = "gpt-oss:20b"):
        self.llm_model = llm_model
        self.console = console
        
    def generate_governance_dataset(self, 
                                  feature_names: List[str],
                                  context: str,
                                  output_size: int = 10000,
                                  output_formats: List[str] = ['csv'],
                                  output_dir: str = "governance_datasets") -> Dict[str, Any]:
        """
        Generate comprehensive governance dataset for AI compliance
        
        Args:
            feature_names: List of feature column names
            context: Domain context (e.g., "loan approval", "fraud detection")
            output_size: Number of rows to generate
            output_formats: Output formats ['csv', 'json', 'parquet', 'governance_report']
            output_dir: Output directory for generated files
            
        Returns:
            Dict with generation results and governance compliance metrics
        """
        
        self.console.print(Panel(
            f"🏛️ [bold blue]AI Governance Dataset Generation[/bold blue]\n"
            f"Features: {len(feature_names)} columns\n"
            f"Context: {context}\n"
            f"Target size: {output_size:,} rows\n"
            f"🎯 Optimized for: ISO 42001, Fairness Testing, Explainability",
            title="AI Governance Pipeline"
        ))
        
        results = {
            'success': False,
            'feature_names': feature_names,
            'context': context,
            'generated_files': [],
            'scenarios_discovered': 0,
            'governance_metrics': {},
            'compliance_summary': {},
            'errors': []
        }
        
        try:
            # Step 1: Generate LLM prompt for governance scenario discovery
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task1 = progress.add_task("🧠 Generating governance-focused LLM prompt...", total=None)
                prompt = self._create_governance_scenario_prompt(feature_names, context)
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                progress.update(task1, completed=True)
            
            # Step 2: Query LLM for governance scenarios
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task2 = progress.add_task("🤖 Querying LLM for governance scenarios...", total=None)
                llm_response = self._query_llm(prompt)
                progress.update(task2, completed=True)
            
            if not llm_response:
                results['errors'].append("Failed to get LLM response")
                return results
            
            # Step 3: Parse LLM response into governance scenarios
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task3 = progress.add_task("📋 Parsing governance scenarios...", total=None)
                scenarios = self._parse_governance_scenarios(llm_response, feature_names, context)
                progress.update(task3, completed=True)
            
            results['scenarios_discovered'] = len(scenarios)
            
            if not scenarios:
                results['errors'].append("No valid governance scenarios extracted from LLM response")
                return results
            
            # Step 4: Generate governance dataset
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task4 = progress.add_task(f"🏛️ Generating governance dataset ({output_size:,} rows)...", total=None)
                dataset, governance_metrics = self._generate_governance_dataset_from_scenarios(
                    scenarios, feature_names, output_size, prompt_hash
                )
                progress.update(task4, completed=True)
            
            results['governance_metrics'] = governance_metrics
            
            # Step 5: Export with governance documentation
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task5 = progress.add_task("💾 Exporting governance dataset...", total=None)
                exported_files = self._export_governance_dataset(
                    dataset, feature_names, context, output_formats, output_dir
                )
                progress.update(task5, completed=True)
            
            results['generated_files'] = exported_files
            results['success'] = True
            
            # Generate compliance summary
            results['compliance_summary'] = self._generate_compliance_summary(dataset)
            
            # Display governance success summary
            self._display_governance_summary(results)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            results['errors'].append(str(e))
            self.console.print(f"[red]❌ Error in governance pipeline: {e}[/red]")
            self.console.print(f"[red]Full traceback:[/red]")
            self.console.print(error_details)
        
        return results
    
    def _create_governance_scenario_prompt(self, feature_names: List[str], context: str) -> str:
        """Create LLM prompt optimized for AI governance scenarios"""
        
        features_str = ", ".join(feature_names)
        
        prompt = f"""Reasoning: high

You are an expert in AI governance, ISO 42001 compliance, fairness testing, and explainable AI.

TASK: Generate comprehensive testing scenarios for AI governance compliance in a {context} model with these features:
{features_str}

CRITICAL REQUIREMENTS for AI Governance:

1. ISO 42001 COMPLIANCE SCENARIOS:
   - Risk management testing (high-risk decisions)
   - Documentation and traceability scenarios
   - Governance process validation
   - Audit trail requirements
   - Quality management scenarios

2. FAIRNESS & BIAS TESTING (EU AI Act Article 15):
   - Protected class combinations (gender, race, age, disability)
   - Intersectional bias testing (multiple protected classes)
   - Historical discrimination pattern detection
   - Demographic parity testing scenarios
   - Equal opportunity assessment cases

3. EXPLAINABILITY SCENARIOS (EU AI Act Article 13):
   - Simple decision explanations
   - Complex multi-factor interactions
   - Counter-intuitive decision scenarios
   - Edge case explanations
   - Stakeholder-specific explanation needs

4. TRANSPARENCY & ACCOUNTABILITY:
   - Decision boundary testing
   - Model behavior documentation
   - Stakeholder communication scenarios
   - Regulatory reporting requirements

5. RISK ASSESSMENT SCENARIOS:
   - High-impact decision testing
   - Safety-critical scenarios
   - Business risk validation
   - Regulatory compliance edge cases

For each scenario, specify:
- Scenario name and governance purpose
- ISO 42001 risk category (low/medium/high/critical)
- Protected classes involved (if any)
- Explainability complexity (simple/moderate/complex/edge_case)
- Compliance requirements (ISO42001, EU AI Act articles)
- Why this scenario is critical for AI governance
- Expected realistic value patterns for governance testing

Focus on {context} domain with AI governance expertise. Generate scenarios that ensure:
✅ ISO 42001 compliance coverage
✅ EU AI Act Article 13 & 15 compliance
✅ Comprehensive bias testing
✅ Explainability validation
✅ Audit trail completeness

Generate at least 20-25 distinct governance scenarios for production-ready AI compliance."""

        return prompt
    
    def _save_llm_interaction(self, prompt: str, response: str):
        """Save LLM prompt and response with governance metadata"""
        
        # Create interactions directory
        interactions_dir = Path("governance_llm_interactions")
        interactions_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save interaction with governance context
        interaction_file = interactions_dir / f"governance_interaction_{timestamp}.txt"
        
        with open(interaction_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"AI GOVERNANCE LLM INTERACTION - {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.llm_model}\n")
            f.write(f"Purpose: AI Governance Dataset Generation\n")
            f.write(f"Compliance Focus: ISO 42001, EU AI Act Articles 13 & 15\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("GOVERNANCE PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt)
            f.write("\n\n")
            
            f.write("LLM RESPONSE:\n")
            f.write("-" * 40 + "\n")
            f.write(response)
            f.write("\n\n")
            
            f.write("GOVERNANCE METADATA:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Prompt Hash: {hashlib.md5(prompt.encode()).hexdigest()}\n")
            f.write(f"Response Length: {len(response)} characters\n")
            f.write(f"Generation Purpose: AI Governance Compliance\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        self.console.print(f"[dim]🏛️ Saved governance LLM interaction to: {interaction_file}[/dim]")
    
    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query LLM using Ollama with governance context"""
        
        try:
            # Create temporary file for prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            # Query Ollama
            cmd = ['ollama', 'run', self.llm_model]
            
            with open(prompt_file, 'r') as f:
                result = subprocess.run(
                    cmd,
                    input=f.read(),
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for governance scenarios
                )
            
            # Cleanup
            os.unlink(prompt_file)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Save governance interaction
                self._save_llm_interaction(prompt, response)
                
                return response
            else:
                self.console.print(f"[red]LLM Error: {result.stderr}[/red]")
                return None
                
        except subprocess.TimeoutExpired:
            self.console.print("[red]LLM query timed out[/red]")
            return None
        except Exception as e:
            self.console.print(f"[red]Error querying LLM: {e}[/red]")
            return None
    
    def _parse_governance_scenarios(self, llm_response: str, feature_names: List[str], context: str) -> List[Dict[str, Any]]:
        """Parse LLM response into governance scenarios"""
        
        scenarios = []
        
        # Enhanced patterns for governance scenarios
        scenario_patterns = [
            r"(?i)scenario\s*\d*[:\-\s]*([^:]+?)[:]*\s*\n([^#\n]+)",
            r"(?i)(\d+\.?\s*[^:\n]+?)[:]*\s*\n([^#\n]+)",
            r"(?i)#{1,3}\s*([^#\n]+)\s*\n([^#]+?)(?=\n#{1,3}|\n\d+\.|\Z)"
        ]
        
        for pattern in scenario_patterns:
            matches = re.findall(pattern, llm_response, re.MULTILINE | re.DOTALL)
            for match in matches:
                name = match[0].strip()
                description = match[1].strip()
                
                if len(name) > 5 and len(description) > 20:  # Basic validation
                    scenario = self._create_governance_scenario_template(name, description, feature_names, context)
                    if scenario:
                        scenarios.append(scenario)
        
        # If pattern matching fails, create default governance scenarios
        if len(scenarios) < 5:
            scenarios.extend(self._create_default_governance_scenarios(feature_names, context))
        
        return scenarios[:25]  # Limit to top 25 governance scenarios
    
    def _create_governance_scenario_template(self, name: str, description: str, 
                                           feature_names: List[str], context: str) -> Optional[Dict[str, Any]]:
        """Create governance scenario template with compliance metadata"""
        
        scenario = {
            'name': name[:50],
            'description': description[:200],
            'template': {},
            'priority': 'medium',
            # Governance additions
            'bias_testing_scenario': False,
            'protected_classes_involved': [],
            'iso42001_risk_category': 'medium',
            'explainability_target': 'standard',
            'compliance_requirements': []
        }
        
        # Analyze for governance characteristics
        content = (name + " " + description).lower()
        
        # Determine if bias testing scenario
        bias_keywords = ['bias', 'discrimination', 'fairness', 'protected', 'demographic', 'gender', 'race', 'age', 'disability']
        scenario['bias_testing_scenario'] = any(keyword in content for keyword in bias_keywords)
        
        # Extract protected classes
        if 'gender' in content or 'male' in content or 'female' in content:
            scenario['protected_classes_involved'].append('gender')
        if 'race' in content or 'racial' in content or 'ethnic' in content:
            scenario['protected_classes_involved'].append('race')
        if 'age' in content or 'senior' in content or 'elderly' in content or 'young' in content:
            scenario['protected_classes_involved'].append('age')
        if 'disability' in content or 'disabled' in content:
            scenario['protected_classes_involved'].append('disability')
        
        # Determine ISO 42001 risk category
        if any(term in content for term in ['critical', 'high risk', 'severe', 'safety']):
            scenario['iso42001_risk_category'] = 'critical'
            scenario['priority'] = 'critical'
        elif any(term in content for term in ['high', 'significant', 'important', 'bias', 'discrimination']):
            scenario['iso42001_risk_category'] = 'high'
            scenario['priority'] = 'high'
        elif any(term in content for term in ['low', 'minor', 'routine']):
            scenario['iso42001_risk_category'] = 'low'
            scenario['priority'] = 'low'
        
        # Determine explainability target
        if any(term in content for term in ['complex', 'interaction', 'non-linear', 'sophisticated']):
            scenario['explainability_target'] = 'complex'
        elif any(term in content for term in ['edge', 'boundary', 'unusual', 'rare']):
            scenario['explainability_target'] = 'edge_case'
        elif any(term in content for term in ['simple', 'basic', 'straightforward']):
            scenario['explainability_target'] = 'simple'
        
        # Determine compliance requirements
        if scenario['bias_testing_scenario']:
            scenario['compliance_requirements'].extend(['ISO42001_Fairness', 'EU_AI_Act_Art15'])
        if any(term in content for term in ['explanation', 'transparency', 'interpretable']):
            scenario['compliance_requirements'].extend(['ISO42001_Transparency', 'EU_AI_Act_Art13'])
        if any(term in content for term in ['audit', 'documentation', 'governance']):
            scenario['compliance_requirements'].extend(['ISO42001_Documentation'])
        if any(term in content for term in ['risk', 'safety', 'critical']):
            scenario['compliance_requirements'].extend(['ISO42001_Risk_Management'])
        
        # Extract constraints for the scenario
        scenario['template'] = self._extract_governance_constraints(content, feature_names, context)
        
        return scenario if scenario['template'] or scenario['bias_testing_scenario'] else None
    
    def _extract_governance_constraints(self, content: str, feature_names: List[str], context: str) -> Dict[str, Any]:
        """Extract constraints specific to governance testing"""
        constraints = {}
        
        # Financial governance constraints for loan context
        if context.lower() in ['loan', 'credit', 'banking', 'finance']:
            if 'high income' in content or 'wealthy' in content:
                constraints['income'] = (100000, 500000)
            elif 'low income' in content or 'poor' in content or 'disadvantaged' in content:
                constraints['income'] = (15000, 40000)
            
            if 'excellent credit' in content or 'high credit' in content:
                constraints['credit_score'] = (750, 850)
            elif 'poor credit' in content or 'bad credit' in content or 'subprime' in content:
                constraints['credit_score'] = (300, 580)
            
            if 'young' in content or 'youth' in content:
                constraints['age'] = (18, 30)
            elif 'senior' in content or 'elderly' in content or 'retirement' in content:
                constraints['age'] = (65, 80)
            
            # Protected class constraints
            if 'female' in content or 'women' in content:
                constraints['gender'] = ['Female']
            elif 'male' in content or 'men' in content:
                constraints['gender'] = ['Male']
            
            if 'black' in content or 'african american' in content:
                constraints['race'] = ['Black']
            elif 'hispanic' in content or 'latino' in content:
                constraints['race'] = ['Hispanic/Latino']
            elif 'asian' in content:
                constraints['race'] = ['Asian']
        
        return constraints
    
    def _create_default_governance_scenarios(self, feature_names: List[str], context: str) -> List[Dict[str, Any]]:
        """Create default governance scenarios if LLM parsing fails"""
        
        return [
            {
                'name': 'Gender Bias Testing - Female Applicants',
                'description': 'Test for gender-based lending discrimination against female applicants',
                'template': {'gender': ['Female']},
                'priority': 'critical',
                'bias_testing_scenario': True,
                'protected_classes_involved': ['gender'],
                'iso42001_risk_category': 'critical',
                'explainability_target': 'standard',
                'compliance_requirements': ['ISO42001_Fairness', 'EU_AI_Act_Art15']
            },
            {
                'name': 'Racial Bias Testing - Minority Groups',
                'description': 'Test for racial discrimination in lending decisions',
                'template': {'race': ['Black', 'Hispanic/Latino']},
                'priority': 'critical',
                'bias_testing_scenario': True,
                'protected_classes_involved': ['race'],
                'iso42001_risk_category': 'critical',
                'explainability_target': 'standard',
                'compliance_requirements': ['ISO42001_Fairness', 'EU_AI_Act_Art15']
            },
            {
                'name': 'Age Discrimination - Senior Citizens',
                'description': 'Test for age-based lending discrimination against seniors',
                'template': {'age': (65, 80)},
                'priority': 'high',
                'bias_testing_scenario': True,
                'protected_classes_involved': ['age'],
                'iso42001_risk_category': 'high',
                'explainability_target': 'standard',
                'compliance_requirements': ['ISO42001_Fairness', 'EU_AI_Act_Art15']
            },
            {
                'name': 'Complex Decision Boundary Analysis',
                'description': 'Edge cases for explainability testing at decision boundaries',
                'template': {'credit_score': (650, 700), 'income': (45000, 55000)},
                'priority': 'medium',
                'bias_testing_scenario': False,
                'protected_classes_involved': [],
                'iso42001_risk_category': 'medium',
                'explainability_target': 'edge_case',
                'compliance_requirements': ['ISO42001_Transparency', 'EU_AI_Act_Art13']
            },
            {
                'name': 'High-Risk Decision Documentation',
                'description': 'High-risk lending scenarios requiring detailed audit trails',
                'template': {'credit_score': (300, 500), 'debt_to_income_ratio': (0.7, 1.2)},
                'priority': 'high',
                'bias_testing_scenario': False,
                'protected_classes_involved': [],
                'iso42001_risk_category': 'high',
                'explainability_target': 'complex',
                'compliance_requirements': ['ISO42001_Risk_Management', 'ISO42001_Documentation']
            }
        ]
    
    def _generate_governance_dataset_from_scenarios(self, scenarios: List[Dict[str, Any]], 
                                                  feature_names: List[str], 
                                                  output_size: int,
                                                  prompt_hash: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate governance dataset from scenarios"""
        
        # Initialize governance generator
        generator = AIGovernanceDatasetGenerator(
            seed=42,
            llm_model=self.llm_model,
            prompt_hash=prompt_hash
        )
        
        # Add LLM scenarios with governance metadata
        generator.add_llm_scenarios(scenarios, {'source': 'governance_llm_analysis'})
        
        # Generate governance dataset with bias testing focus
        dataset = generator.generate_dataset(
            total_rows=output_size, 
            bias_testing_ratio=0.4  # 40% bias testing for compliance
        )
        
        # Filter to requested features if they exist
        available_features = [f for f in feature_names if f in dataset.columns]
        if available_features and len(available_features) == len(feature_names):
            # Keep governance columns + requested features
            governance_cols = [col for col in dataset.columns if col.startswith('gov_')]
            dataset = dataset[available_features + governance_cols]
        elif available_features:
            missing_features = set(feature_names) - set(available_features)
            print(f"Note: Some requested features not generated: {missing_features}")
            print(f"Generated dataset includes: {list(dataset.columns)}")
        
        # Generate governance metrics
        governance_metrics = {
            'total_records': len(dataset),
            'bias_testing_records': dataset['gov_bias_testing_flag'].sum() if 'gov_bias_testing_flag' in dataset.columns else 0,
            'critical_risk_records': len(dataset[dataset['gov_risk_category'] == 'critical']) if 'gov_risk_category' in dataset.columns else 0,
            'compliance_coverage': self._calculate_compliance_coverage(dataset),
            'protected_class_diversity': self._calculate_protected_class_diversity(dataset)
        }
        
        return dataset, governance_metrics
    
    def _calculate_compliance_coverage(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Calculate compliance coverage metrics"""
        coverage = {}
        
        if 'gov_compliance_tags' in dataset.columns:
            all_tags = []
            for tags_str in dataset['gov_compliance_tags'].dropna():
                all_tags.extend(tags_str.split(','))
            
            unique_tags = set(tag.strip() for tag in all_tags if tag.strip())
            coverage = {
                'total_compliance_requirements': len(unique_tags),
                'iso42001_coverage': len([tag for tag in unique_tags if 'ISO42001' in tag]),
                'eu_ai_act_coverage': len([tag for tag in unique_tags if 'EU_AI_Act' in tag]),
                'requirements_list': list(unique_tags)
            }
        
        return coverage
    
    def _calculate_protected_class_diversity(self, dataset: pd.DataFrame) -> Dict[str, int]:
        """Calculate protected class diversity for bias testing"""
        diversity = {}
        
        protected_classes = ['gender', 'race', 'age_group', 'disability_status']
        for pc in protected_classes:
            if pc in dataset.columns:
                diversity[f'{pc}_diversity'] = len(dataset[pc].unique())
        
        return diversity
    
    def _export_governance_dataset(self, dataset: pd.DataFrame, feature_names: List[str], 
                                 context: str, output_formats: List[str], output_dir: str) -> List[str]:
        """Export governance dataset with compliance documentation"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate filename with governance context
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"governance_{context.replace(' ', '_')}_{timestamp}"
        
        exported_files = []
        
        # Always include governance report
        if 'governance_report' not in output_formats:
            output_formats.append('governance_report')
        
        for fmt in output_formats:
            if fmt == 'csv':
                filepath = Path(output_dir) / f"{base_filename}.csv"
                dataset.to_csv(filepath, index=False)
            elif fmt == 'json':
                filepath = Path(output_dir) / f"{base_filename}.json"
                dataset.to_json(filepath, orient='records', indent=2)
            elif fmt == 'parquet':
                filepath = Path(output_dir) / f"{base_filename}.parquet"
                dataset.to_parquet(filepath, index=False)
            elif fmt == 'governance_report':
                filepath = Path(output_dir) / f"{base_filename}_compliance_report.json"
                self._generate_comprehensive_governance_report(dataset, filepath)
            
            exported_files.append(str(filepath))
        
        return exported_files
    
    def _generate_comprehensive_governance_report(self, dataset: pd.DataFrame, filepath: Path):
        """Generate comprehensive governance compliance report"""
        
        governance_cols = [col for col in dataset.columns if col.startswith('gov_')]
        
        report = {
            'governance_summary': {
                'generation_timestamp': datetime.now().isoformat(),
                'llm_model_used': self.llm_model,
                'total_records': len(dataset),
                'governance_fields': len(governance_cols),
                'compliance_framework': 'ISO 42001, EU AI Act Articles 13 & 15'
            },
            'iso42001_compliance': {
                'risk_management': {
                    'total_records': len(dataset),
                    'high_risk_records': len(dataset[dataset['gov_risk_category'].isin(['high', 'critical'])]) if 'gov_risk_category' in dataset.columns else 0,
                    'risk_distribution': dataset['gov_risk_category'].value_counts().to_dict() if 'gov_risk_category' in dataset.columns else {}
                },
                'fairness_testing': {
                    'bias_testing_records': dataset['gov_bias_testing_flag'].sum() if 'gov_bias_testing_flag' in dataset.columns else 0,
                    'bias_testing_percentage': f"{(dataset['gov_bias_testing_flag'].sum() / len(dataset) * 100):.1f}%" if 'gov_bias_testing_flag' in dataset.columns else "0%",
                    'protected_classes_covered': len(dataset['gov_protected_classes'].unique()) if 'gov_protected_classes' in dataset.columns else 0
                },
                'transparency': {
                    'explainability_records': len(dataset[dataset['gov_explainability_complexity'] != 'simple']) if 'gov_explainability_complexity' in dataset.columns else 0,
                    'complexity_distribution': dataset['gov_explainability_complexity'].value_counts().to_dict() if 'gov_explainability_complexity' in dataset.columns else {}
                }
            },
            'eu_ai_act_compliance': {
                'article_13_transparency': {
                    'explanation_ready_records': len(dataset[dataset['gov_explainability_complexity'].notna()]) if 'gov_explainability_complexity' in dataset.columns else 0,
                    'decision_boundary_cases': dataset['gov_decision_boundary_flag'].sum() if 'gov_decision_boundary_flag' in dataset.columns else 0
                },
                'article_15_fairness': {
                    'bias_testing_coverage': dataset['gov_bias_testing_flag'].sum() if 'gov_bias_testing_flag' in dataset.columns else 0,
                    'protected_class_scenarios': len(dataset[dataset['gov_protected_classes'].str.len() > 0]) if 'gov_protected_classes' in dataset.columns else 0
                }
            },
            'data_quality_metrics': {
                'completeness': {
                    'total_fields': len(dataset.columns),
                    'governance_fields': len(governance_cols),
                    'missing_values': dataset.isnull().sum().sum()
                },
                'diversity': self._calculate_protected_class_diversity(dataset),
                'audit_trail': {
                    'all_records_traced': len(dataset[dataset['gov_audit_trail_id'].notna()]) if 'gov_audit_trail_id' in dataset.columns else 0,
                    'unique_scenarios': len(dataset['gov_scenario_source'].unique()) if 'gov_scenario_source' in dataset.columns else 0
                }
            },
            'recommendations': {
                'iso42001': [
                    "Implement regular bias testing using the provided bias_testing_flag records",
                    "Use risk_category distribution for risk-based testing strategies",
                    "Leverage audit_trail_id for complete traceability documentation"
                ],
                'fairness_testing': [
                    "Focus on records with bias_testing_flag=True for fairness analysis",
                    "Test all protected_class combinations for intersectional bias",
                    "Monitor decision outcomes across demographic groups"
                ],
                'explainability': [
                    "Use explainability_complexity field to prioritize explanation efforts",
                    "Focus on edge_case scenarios for boundary explanation testing",
                    "Implement different explanation strategies by complexity level"
                ]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _generate_compliance_summary(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level compliance summary"""
        return {
            'iso42001_ready': len(dataset[dataset['gov_risk_category'].notna()]) > 0 if 'gov_risk_category' in dataset.columns else False,
            'bias_testing_ready': dataset['gov_bias_testing_flag'].sum() > 0 if 'gov_bias_testing_flag' in dataset.columns else False,
            'explainability_ready': len(dataset[dataset['gov_explainability_complexity'].notna()]) > 0 if 'gov_explainability_complexity' in dataset.columns else False,
            'audit_trail_complete': len(dataset[dataset['gov_audit_trail_id'].notna()]) == len(dataset) if 'gov_audit_trail_id' in dataset.columns else False
        }
    
    def _display_governance_summary(self, results: Dict[str, Any]):
        """Display governance generation success summary"""
        
        # Governance success panel
        governance_text = f"""
🏛️ [green]AI Governance Dataset Generated![/green]

📊 **Dataset Metrics:**
• Total Records: {results['governance_metrics'].get('total_records', 'N/A'):,}
• Bias Testing Records: {results['governance_metrics'].get('bias_testing_records', 'N/A'):,}
• Critical Risk Records: {results['governance_metrics'].get('critical_risk_records', 'N/A'):,}

🎯 **Compliance Coverage:**
• ISO 42001: {'✅ Ready' if results['compliance_summary'].get('iso42001_ready') else '❌ Not Ready'}
• Bias Testing: {'✅ Ready' if results['compliance_summary'].get('bias_testing_ready') else '❌ Not Ready'}
• Explainability: {'✅ Ready' if results['compliance_summary'].get('explainability_ready') else '❌ Not Ready'}
• Audit Trail: {'✅ Complete' if results['compliance_summary'].get('audit_trail_complete') else '❌ Incomplete'}

🧠 **LLM Analysis:**
• Governance Scenarios: {results['scenarios_discovered']}
• Domain Context: {results['context']}

💾 **Generated Files:**
{chr(10).join(f"• {Path(f).name}" for f in results['generated_files'])}
        """
        
        self.console.print(Panel(governance_text, title="🏛️ AI Governance Success", border_style="green"))
        
        # Protected class diversity table
        if 'protected_class_diversity' in results['governance_metrics']:
            diversity_table = Table(title="Protected Class Diversity (Bias Testing)")
            diversity_table.add_column("Protected Class", style="cyan")
            diversity_table.add_column("Unique Values", style="white")
            
            for pc, count in results['governance_metrics']['protected_class_diversity'].items():
                diversity_table.add_row(pc.replace('_diversity', '').title(), f"{count}")
            
            self.console.print(diversity_table)

def main():
    """Main CLI interface for AI governance dataset generation"""
    parser = argparse.ArgumentParser(
        description="AI Governance Dataset Generation CLI - Generate datasets for ISO 42001, fairness testing, and explainability"
    )
    
    parser.add_argument(
        'features',
        nargs='+',
        help='Feature names (space-separated)'
    )
    
    parser.add_argument(
        '--context',
        required=True,
        help='Domain context (e.g., "loan approval", "fraud detection", "hiring")'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=10000,
        help='Dataset size (default: 10000)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['csv', 'json', 'parquet', 'governance_report'],
        default=['csv', 'governance_report'],
        help='Output formats (default: csv, governance_report)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='governance_datasets',
        help='Output directory (default: governance_datasets)'
    )
    
    parser.add_argument(
        '--llm-model',
        default='gpt-oss:20b',
        help='LLM model to use (default: gpt-oss:20b)'
    )
    
    args = parser.parse_args()
    
    # Initialize governance CLI
    cli = AIGovernanceDatasetCLI(llm_model=args.llm_model)
    
    # Display input summary with governance focus
    console.print(Panel(
        f"🎯 **Features:** {', '.join(args.features[:5])}{'...' if len(args.features) > 5 else ''}\n"
        f"🎯 **Context:** {args.context}\n"
        f"🎯 **Target Size:** {args.size:,} rows\n"
        f"🎯 **Formats:** {', '.join(args.formats)}\n"
        f"🏛️ **Governance Focus:** ISO 42001, EU AI Act Compliance",
        title="🏛️ AI Governance Dataset Generation",
        border_style="blue"
    ))
    
    # Generate governance dataset
    results = cli.generate_governance_dataset(
        feature_names=args.features,
        context=args.context,
        output_size=args.size,
        output_formats=args.formats,
        output_dir=args.output_dir
    )
    
    if results['success']:
        console.print(f"\n[green]🏛️ Success! Governance datasets generated in: {args.output_dir}[/green]")
        console.print(f"[green]📋 Compliance report: *_compliance_report.json[/green]")
        sys.exit(0)
    else:
        console.print(f"\n[red]❌ Governance generation failed: {', '.join(results['errors'])}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()