#!/usr/bin/env python3
"""
AI Dataset Generation CLI

Generic AI Dataset Generator using reusable LLM core services for dynamic model
selection and integration to synthesize structured datasets from user-provided context prompts.

Features:
- AWS LLM Core Services integration with dynamic model selection
- MLOps pipelines featuring RAG, fine-tuning, and guardrails
- Automated evaluation strategies for data integrity validation
- Comprehensive test coverage for bias testing and compliance
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

# Import AWS LLM Core Services and MLOps Pipeline
from aws_llm_services import LLMCoreService, create_llm_service, AWSConfig, ModelConfig
from mlops_pipeline import (
    MLOpsPipeline, 
    create_mlops_pipeline,
    RAGPipeline,
    GuardrailsPipeline,
    AutomatedEvaluationPipeline
)

console = Console()

class AIDatasetCLI:
    """
    AI Dataset Generation CLI for comprehensive test data.
    
    Integrates AWS LLM Core Services for dynamic model selection and
    MLOps pipelines featuring RAG, fine-tuning, guardrails, and automated evaluation.
    """
    
    def __init__(self, llm_model: str = "gpt-oss:20b", aws_region: str = "us-east-1",
                 enable_rag: bool = True, enable_guardrails: bool = True):
        self.llm_model = llm_model
        self.console = console
        
        # Initialize AWS LLM Core Services
        self.llm_service = create_llm_service(
            aws_region=aws_region,
            default_model=llm_model,
            enable_bedrock=True,
            enable_sagemaker=False
        )
        
        # Initialize MLOps Pipeline
        self.mlops_pipeline = None
        self.enable_rag = enable_rag
        self.enable_guardrails = enable_guardrails
        
    def generate_comprehensive_dataset(self, 
                                     feature_names: List[str],
                                     context: str,
                                     output_size: int = 10000,
                                     output_formats: List[str] = ['csv'],
                                     output_dir: str = "datasets") -> Dict[str, Any]:
        """
        Generate comprehensive test dataset for downstream AI analysis.
        
        Uses AWS LLM Core Services with dynamic model selection and MLOps pipelines
        featuring RAG, guardrails, and automated evaluation for high-quality outputs.
        
        Args:
            feature_names: List of feature column names
            context: Domain context (e.g., "loan approval", "fraud detection")
            output_size: Number of rows to generate
            output_formats: Output formats ['csv', 'json', 'parquet']
            output_dir: Output directory for generated files
            
        Returns:
            Dict with generation results and dataset statistics
        """
        
        # Initialize MLOps pipeline for this context
        self._initialize_mlops_pipeline(context)
        
        self.console.print(Panel(
            f"[bold blue]AI Dataset Generation[/bold blue]\n"
            f"Features: {len(feature_names)} columns\n"
            f"Context: {context}\n"
            f"Target size: {output_size:,} rows\n"
            f"LLM Model: {self.llm_model}\n"
            f"RAG Enabled: {self.enable_rag}\n"
            f"Guardrails Enabled: {self.enable_guardrails}\n"
            f"Comprehensive test data with MLOps pipeline",
            title="Dataset Generation Pipeline"
        ))
        
        results = {
            'success': False,
            'feature_names': feature_names,
            'context': context,
            'generated_files': [],
            'scenarios_discovered': 0,
            'dataset_stats': {},
            'coverage_metrics': {},
            'mlops_metrics': {},
            'guardrail_results': {},
            'evaluation_results': {},
            'errors': []
        }
        
        try:
            # Step 1: Generate LLM prompt with RAG enhancement
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task1 = progress.add_task("Generating RAG-enhanced scenario prompt...", total=None)
                prompt, rag_metadata = self._create_rag_enhanced_prompt(feature_names, context)
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                results['mlops_metrics']['rag'] = rag_metadata
                progress.update(task1, completed=True)
            
            # Step 2: Apply input guardrails
            if self.enable_guardrails and self.mlops_pipeline:
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                    task_guard = progress.add_task("Applying input guardrails...", total=None)
                    guardrail_result = self.mlops_pipeline.guardrails.check_input(prompt)
                    results['guardrail_results']['input'] = guardrail_result.to_dict()
                    if not guardrail_result.passed:
                        results['errors'].append("Input blocked by guardrails")
                        return results
                    progress.update(task_guard, completed=True)
            
            # Step 3: Query LLM using AWS Core Services with dynamic model selection
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task2 = progress.add_task("Querying LLM for test scenarios...", total=None)
                
                # Use dynamic model selection
                optimal_model = self.llm_service.select_optimal_model(
                    task_type="analysis",
                    prefer_local=True
                )
                self.console.print(f"[dim]Selected model: {optimal_model}[/dim]")
                
                llm_response = self._query_llm_with_service(prompt, optimal_model)
                progress.update(task2, completed=True)
            
            if not llm_response:
                # Fallback to direct Ollama query
                llm_response = self._query_llm(prompt)
            
            if not llm_response:
                results['errors'].append("Failed to get LLM response")
                return results
            
            # Step 4: Apply output guardrails
            if self.enable_guardrails and self.mlops_pipeline:
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                    task_guard_out = progress.add_task("Applying output guardrails...", total=None)
                    output_guardrail = self.mlops_pipeline.guardrails.check_output(llm_response)
                    results['guardrail_results']['output'] = output_guardrail.to_dict()
                    if output_guardrail.sanitized_content:
                        llm_response = output_guardrail.sanitized_content
                    progress.update(task_guard_out, completed=True)
            
            # Step 5: Parse LLM response into test scenarios
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task3 = progress.add_task("Parsing test scenarios...", total=None)
                scenarios = self._parse_test_scenarios(llm_response, feature_names, context)
                progress.update(task3, completed=True)
            
            results['scenarios_discovered'] = len(scenarios)
            
            if not scenarios:
                results['errors'].append("No valid scenarios extracted from LLM response")
                return results
            
            # Step 6: Generate comprehensive test dataset
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task4 = progress.add_task(f"Generating test dataset ({output_size:,} rows)...", total=None)
                dataset, dataset_metrics = self._generate_test_dataset_from_scenarios(
                    scenarios, feature_names, output_size, prompt_hash
                )
                progress.update(task4, completed=True)
            
            results['dataset_stats'] = dataset_metrics
            
            # Step 7: Automated evaluation of generated dataset
            if self.mlops_pipeline:
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                    task_eval = progress.add_task("Running automated evaluation...", total=None)
                    eval_results = self._run_automated_evaluation(dataset, feature_names)
                    results['evaluation_results'] = eval_results
                    progress.update(task_eval, completed=True)
            
            # Step 8: Export dataset
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task5 = progress.add_task("Exporting dataset...", total=None)
                exported_files = self._export_test_dataset(
                    dataset, feature_names, context, output_formats, output_dir
                )
                progress.update(task5, completed=True)
            
            results['generated_files'] = exported_files
            results['success'] = True
            
            # Generate coverage metrics
            results['coverage_metrics'] = self._generate_coverage_metrics(dataset)
            
            # Add MLOps pipeline metrics
            if self.mlops_pipeline:
                results['mlops_metrics']['pipeline'] = self.mlops_pipeline.get_pipeline_metrics()
            
            # Add LLM service metrics
            results['mlops_metrics']['llm_service'] = self.llm_service.get_metrics()
            
            # Display success summary
            self._display_success_summary(results)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            results['errors'].append(str(e))
            self.console.print(f"[red]Error in pipeline: {e}[/red]")
            self.console.print(f"[red]Full traceback:[/red]")
            self.console.print(error_details)
        
        return results
    
    def _initialize_mlops_pipeline(self, context: str):
        """Initialize MLOps pipeline with RAG, guardrails, and evaluation for the given context."""
        # Map context to domain
        domain_mapping = {
            'loan': 'loan_approval',
            'credit': 'loan_approval',
            'fraud': 'fraud_detection',
            'hiring': 'hiring',
            'employment': 'hiring'
        }
        
        domain = 'loan_approval'  # Default
        for key, value in domain_mapping.items():
            if key in context.lower():
                domain = value
                break
        
        self.mlops_pipeline = create_mlops_pipeline(
            llm_service=self.llm_service,
            domain=domain
        )
        self.console.print(f"[dim]MLOps pipeline initialized for domain: {domain}[/dim]")
    
    def _create_rag_enhanced_prompt(self, feature_names: List[str], context: str) -> tuple:
        """Create LLM prompt with RAG enhancement for domain-specific context."""
        base_prompt = self._create_comprehensive_scenario_prompt(feature_names, context)
        
        if self.enable_rag and self.mlops_pipeline:
            # Get RAG-enhanced context
            enhanced_prompt, rag_metadata = self.mlops_pipeline.rag.generate_with_context(
                query=f"{context} {' '.join(feature_names)}",
                base_prompt=base_prompt
            )
            return enhanced_prompt, rag_metadata
        
        return base_prompt, {"rag_enabled": False}
    
    def _query_llm_with_service(self, prompt: str, model: str = None) -> Optional[str]:
        """Query LLM using AWS LLM Core Service with dynamic model selection."""
        try:
            response = self.llm_service.generate(
                prompt=prompt,
                model=model,
                temperature=0.7
            )
            
            if response and response.content:
                # Save interaction for audit trail
                self._save_llm_interaction(prompt, response.content)
                return response.content
            
        except Exception as e:
            self.console.print(f"[yellow]LLM service query failed: {e}, falling back to direct query[/yellow]")
        
        return None
    
    def _run_automated_evaluation(self, dataset: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """Run automated evaluation on generated dataset."""
        if not self.mlops_pipeline:
            return {}
        
        # Convert DataFrame to dict format for evaluation
        data_dict = {col: dataset[col].tolist() for col in dataset.columns}
        
        # Create schema from feature names
        schema = {}
        for col in dataset.columns:
            if dataset[col].dtype in ['int64', 'int32']:
                schema[col] = 'int'
            elif dataset[col].dtype in ['float64', 'float32']:
                schema[col] = 'float'
            else:
                schema[col] = 'str'
        
        # Run evaluation
        eval_results = self.mlops_pipeline.evaluate_generated_dataset(data_dict, schema)
        
        return eval_results

    def _create_comprehensive_scenario_prompt(self, feature_names: List[str], context: str) -> str:
        """Create LLM prompt for comprehensive test scenario discovery"""
        
        features_str = ", ".join(feature_names)
        
        prompt = f"""You are an expert in {context} systems and comprehensive ML model testing.

TASK: Generate ALL possible testing scenarios for a comprehensive {context} model with these features:
{features_str}

REQUIREMENTS: Create scenarios covering:

1. DEMOGRAPHIC COMBINATIONS (for comprehensive testing):
   - All age groups × education levels × employment types
   - Gender, race, age, disability combinations
   - Geographic and socioeconomic variations
   - Intersectional combinations (multiple demographics)

2. DOMAIN-SPECIFIC PROFILES:
   - Financial ranges: Very low, Low, Medium, High, Very high
   - Risk patterns: No history, Poor, Fair, Good, Excellent
   - Behavioral scenarios: Different usage patterns
   - Status variations: New, established, inactive accounts

3. EDGE CASES & STRESS TESTING:
   - Boundary value scenarios (min/max values)
   - Unusual but valid combinations
   - Contradictory pattern cases
   - Rare but important situations
   - Mathematical edge cases

4. BIAS TESTING SCENARIOS:
   - Same qualifications across different demographics
   - Historical patterns that might reveal bias
   - Protected class combinations
   - Intersectional bias scenarios

5. DECISION BOUNDARY TESTING:
   - Cases near approval/denial boundaries
   - Complex multi-factor interactions
   - Non-linear decision patterns
   - Counter-intuitive scenarios

For each scenario, provide:
- Scenario name and description
- Why it's important for testing
- Which features are most relevant
- Expected realistic value patterns
- What this scenario might reveal about model behavior

Focus on {context} domain expertise. Be exhaustive - generate scenarios that ensure comprehensive test coverage for any downstream analysis tool.

Generate at least 20-25 distinct scenarios that together provide complete test coverage for bias testing, stress testing, and model validation."""

        return prompt
    
    def _save_llm_interaction(self, prompt: str, response: str):
        """Save LLM prompt and response for transparency"""
        
        # Create interactions directory
        interactions_dir = Path("llm_interactions")
        interactions_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save interaction
        interaction_file = interactions_dir / f"interaction_{timestamp}.txt"
        
        with open(interaction_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"AI DATASET GENERATION - {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.llm_model}\n")
            f.write(f"Purpose: Comprehensive Test Dataset Generation\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt)
            f.write("\n\n")
            
            f.write("RESPONSE:\n")
            f.write("-" * 40 + "\n")
            f.write(response)
            f.write("\n\n")
            
            f.write("METADATA:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Prompt Hash: {hashlib.md5(prompt.encode()).hexdigest()}\n")
            f.write(f"Response Length: {len(response)} characters\n")
            f.write(f"Generation Purpose: Comprehensive Test Data\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        self.console.print(f"[dim]Saved LLM interaction to: {interaction_file}[/dim]")
    
    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query LLM using Ollama"""
        
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
                    timeout=600  # 10 minute timeout
                )
            
            # Cleanup
            os.unlink(prompt_file)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Save interaction
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
    
    def _parse_test_scenarios(self, llm_response: str, feature_names: List[str], context: str) -> List[Dict[str, Any]]:
        """Parse LLM response into test scenarios"""
        
        scenarios = []
        
        # Extract scenarios using pattern matching
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
                    scenario = self._create_test_scenario_template(name, description, feature_names, context)
                    if scenario:
                        scenarios.append(scenario)
        
        # If pattern matching fails, create default scenarios
        if len(scenarios) < 5:
            scenarios.extend(self._create_default_test_scenarios(feature_names, context))
        
        return scenarios[:25]  # Limit to top 25 scenarios
    
    def _create_test_scenario_template(self, name: str, description: str, 
                                     feature_names: List[str], context: str) -> Optional[Dict[str, Any]]:
        """Create test scenario template"""
        
        scenario = {
            'name': name[:50],
            'description': description[:200],
            'template': {},
            'priority': 'medium',
            'bias_testing_scenario': False,
            'protected_classes_involved': [],
            'complexity': 'standard'
        }
        
        # Analyze scenario content
        content = (name + " " + description).lower()
        
        # Determine if bias testing scenario
        bias_keywords = ['bias', 'discrimination', 'fairness', 'protected', 'demographic', 'gender', 'race', 'age', 'disability']
        scenario['bias_testing_scenario'] = any(keyword in content for keyword in bias_keywords)
        
        # Extract protected classes
        if 'gender' in content:
            scenario['protected_classes_involved'].append('gender')
        if 'race' in content or 'racial' in content:
            scenario['protected_classes_involved'].append('race')
        if 'age' in content:
            scenario['protected_classes_involved'].append('age')
        if 'disability' in content:
            scenario['protected_classes_involved'].append('disability')
        
        # Determine complexity
        if any(term in content for term in ['complex', 'interaction', 'non-linear']):
            scenario['complexity'] = 'complex'
        elif any(term in content for term in ['edge', 'boundary', 'unusual']):
            scenario['complexity'] = 'edge_case'
        
        # Extract constraints
        scenario['template'] = self._extract_scenario_constraints(content, feature_names, context)
        
        return scenario if scenario['template'] or scenario['bias_testing_scenario'] else None
    
    def _extract_scenario_constraints(self, content: str, feature_names: List[str], context: str) -> Dict[str, Any]:
        """Extract constraints for test scenarios"""
        constraints = {}
        
        # Domain-specific constraints
        if context.lower() in ['loan', 'credit', 'banking', 'finance']:
            if 'high income' in content:
                constraints['income'] = (100000, 500000)
            elif 'low income' in content:
                constraints['income'] = (15000, 40000)
            
            if 'excellent credit' in content:
                constraints['credit_score'] = (750, 850)
            elif 'poor credit' in content:
                constraints['credit_score'] = (300, 580)
            
            if 'young' in content:
                constraints['age'] = (18, 30)
            elif 'senior' in content:
                constraints['age'] = (65, 80)
            
            # Protected class constraints
            if 'female' in content:
                constraints['gender'] = ['Female']
            if 'black' in content:
                constraints['race'] = ['Black']
        
        return constraints
    
    def _create_default_test_scenarios(self, feature_names: List[str], context: str) -> List[Dict[str, Any]]:
        """Create default test scenarios if LLM parsing fails"""
        
        return [
            {
                'name': 'High Value Profile',
                'description': 'High-value customers with excellent metrics',
                'template': {'income': (80000, 200000), 'age': (30, 50)},
                'priority': 'high',
                'bias_testing_scenario': False,
                'protected_classes_involved': [],
                'complexity': 'standard'
            },
            {
                'name': 'Bias Testing - Demographics',
                'description': 'Test across different demographic groups',
                'template': {'gender': ['Female'], 'race': ['Black', 'Hispanic/Latino']},
                'priority': 'high',
                'bias_testing_scenario': True,
                'protected_classes_involved': ['gender', 'race'],
                'complexity': 'standard'
            },
            {
                'name': 'Edge Cases',
                'description': 'Unusual but valid combinations',
                'template': {'age': (18, 25), 'income': (100000, 300000)},
                'priority': 'medium',
                'bias_testing_scenario': False,
                'protected_classes_involved': [],
                'complexity': 'edge_case'
            }
        ]
    
    def _generate_test_dataset_from_scenarios(self, scenarios: List[Dict[str, Any]], 
                                            feature_names: List[str], 
                                            output_size: int,
                                            prompt_hash: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate test dataset from scenarios"""
        
        # Initialize generator
        generator = AIGovernanceDatasetGenerator(
            seed=42,
            llm_model=self.llm_model,
            prompt_hash=prompt_hash
        )
        
        # Add scenarios
        generator.add_llm_scenarios(scenarios, {'source': 'test_scenario_analysis'})
        
        # Generate dataset with comprehensive coverage
        dataset = generator.generate_dataset(
            total_rows=output_size, 
            bias_testing_ratio=0.4  # 40% for bias testing
        )
        
        # Filter to requested features if they exist
        available_features = [f for f in feature_names if f in dataset.columns]
        if available_features and len(available_features) == len(feature_names):
            # Keep only requested features (remove governance metadata for simplicity)
            dataset = dataset[available_features]
        elif available_features:
            missing_features = set(feature_names) - set(available_features)
            print(f"Note: Some requested features not generated: {missing_features}")
            print(f"Generated dataset includes: {list(dataset.columns)}")
        
        # Generate dataset metrics
        dataset_metrics = {
            'total_rows': len(dataset),
            'total_columns': len(dataset.columns),
            'scenarios_used': len(scenarios),
            'feature_coverage': len(available_features) / len(feature_names) if feature_names else 0
        }
        
        return dataset, dataset_metrics
    
    def _export_test_dataset(self, dataset: pd.DataFrame, feature_names: List[str], 
                           context: str, output_formats: List[str], output_dir: str) -> List[str]:
        """Export test dataset"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{context.replace(' ', '_')}_{timestamp}"
        
        exported_files = []
        
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
            
            exported_files.append(str(filepath))
        
        return exported_files
    
    def _generate_coverage_metrics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Generate test coverage metrics"""
        
        coverage = {
            'demographic_diversity': {},
            'value_ranges': {},
            'scenario_coverage': 'comprehensive'
        }
        
        # Calculate demographic diversity
        demographic_cols = ['gender', 'race', 'age_group']
        for col in demographic_cols:
            if col in dataset.columns:
                coverage['demographic_diversity'][col] = len(dataset[col].unique())
        
        # Calculate value ranges for numeric columns
        numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            coverage['value_ranges'][col] = {
                'min': float(dataset[col].min()),
                'max': float(dataset[col].max()),
                'mean': float(dataset[col].mean())
            }
        
        return coverage
    
    def _display_success_summary(self, results: Dict[str, Any]):
        """Display generation success summary"""
        
        # Success panel
        success_text = f"""
[green]Test Dataset Generated![/green]

**Dataset Metrics:**
• Total Records: {results['dataset_stats'].get('total_rows', 'N/A'):,}
• Total Columns: {results['dataset_stats'].get('total_columns', 'N/A')}
• Test Scenarios: {results['scenarios_discovered']}

**Coverage:**
• Feature Coverage: {results['dataset_stats'].get('feature_coverage', 0):.1%}
• Comprehensive test scenarios discovered by LLM
• Ready for downstream analysis tools

**LLM Analysis:**
• Domain Context: {results['context']}
• Comprehensive scenario discovery completed

**Generated Files:**
{chr(10).join(f"• {Path(f).name}" for f in results['generated_files'])}
        """
        
        self.console.print(Panel(success_text, title="Success", border_style="green"))
        
        # Coverage metrics table
        if 'coverage_metrics' in results and 'demographic_diversity' in results['coverage_metrics']:
            diversity_table = Table(title="Test Coverage Diversity")
            diversity_table.add_column("Dimension", style="cyan")
            diversity_table.add_column("Unique Values", style="white")
            
            for dimension, count in results['coverage_metrics']['demographic_diversity'].items():
                diversity_table.add_row(dimension.title(), f"{count}")
            
            if diversity_table.row_count > 0:
                self.console.print(diversity_table)

def main():
    """Main CLI interface for comprehensive test dataset generation"""
    parser = argparse.ArgumentParser(
        description="AI Dataset Generator - Generate comprehensive test datasets for downstream AI analysis"
    )
    
    parser.add_argument(
        'features',
        nargs='+',
        help='Feature names (space-separated)'
    )
    
    parser.add_argument(
        '--context',
        required=True,
        help='Domain context (e.g., "loan approval", "fraud detection", "hiring decisions")'
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
        choices=['csv', 'json', 'parquet'],
        default=['csv'],
        help='Output formats (default: csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='datasets',
        help='Output directory (default: datasets)'
    )
    
    parser.add_argument(
        '--llm-model',
        default='gpt-oss:20b',
        help='LLM model to use (default: gpt-oss:20b)'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = AIDatasetCLI(llm_model=args.llm_model)
    
    # Display input summary
    console.print(Panel(
        f"**Features:** {', '.join(args.features[:5])}{'...' if len(args.features) > 5 else ''}\n"
        f"**Context:** {args.context}\n"
        f"**Target Size:** {args.size:,} rows\n"
        f"**Formats:** {', '.join(args.formats)}\n"
        f"**Purpose:** Comprehensive test data for downstream analysis",
        title="AI Dataset Generation",
        border_style="blue"
    ))
    
    # Generate dataset
    results = cli.generate_comprehensive_dataset(
        feature_names=args.features,
        context=args.context,
        output_size=args.size,
        output_formats=args.formats,
        output_dir=args.output_dir
    )
    
    if results['success']:
        console.print(f"\n[green]Success! Test dataset generated in: {args.output_dir}[/green]")
        console.print(f"[green]Ready for downstream analysis tools![/green]")
        sys.exit(0)
    else:
        console.print(f"\n[red]Generation failed: {', '.join(results['errors'])}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()