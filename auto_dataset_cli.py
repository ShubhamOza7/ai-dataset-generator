#!/usr/bin/env python3
"""
Automated Dataset Generation CLI
Takes feature names → Generates comprehensive testing dataset automatically
"""

import argparse
import json
import subprocess
import sys
import os
import tempfile
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Import our generator
from loan_dataset_generator import LoanDatasetGenerator, LoanScenario

console = Console()

class AutoDatasetCLI:
    """Automated dataset generation CLI"""
    
    def __init__(self, llm_model: str = "gpt-oss:20b"):
        self.llm_model = llm_model
        self.console = console
        
    def generate_comprehensive_dataset(self, 
                                     feature_names: List[str],
                                     context: str,
                                     output_size: int = 10000,
                                     output_formats: List[str] = ['csv'],
                                     output_dir: str = "generated_datasets") -> Dict[str, Any]:
        """
        Fully automated dataset generation pipeline
        
        Args:
            feature_names: List of feature column names
            context: Domain context (e.g., "loan approval", "fraud detection")
            output_size: Number of rows to generate
            output_formats: Output formats ['csv', 'json', 'parquet', 'excel']
            output_dir: Output directory for generated files
            
        Returns:
            Dict with generation results and file paths
        """
        
        self.console.print(Panel(
            f"🚀 [bold blue]Automated Dataset Generation[/bold blue]\n"
            f"Features: {len(feature_names)} columns\n"
            f"Context: {context}\n"
            f"Target size: {output_size:,} rows",
            title="Starting Generation Pipeline"
        ))
        
        results = {
            'success': False,
            'feature_names': feature_names,
            'context': context,
            'generated_files': [],
            'scenarios_discovered': 0,
            'dataset_stats': {},
            'errors': []
        }
        
        try:
            # Step 1: Generate LLM prompt for scenario discovery
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task1 = progress.add_task("🧠 Generating LLM prompt...", total=None)
                prompt = self._create_scenario_discovery_prompt(feature_names, context)
                progress.update(task1, completed=True)
            
            # Step 2: Query LLM for comprehensive scenarios
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task2 = progress.add_task("🤖 Querying LLM for scenarios...", total=None)
                llm_response = self._query_llm(prompt)
                progress.update(task2, completed=True)
            
            if not llm_response:
                results['errors'].append("Failed to get LLM response")
                return results
            
            # Step 3: Parse LLM response into structured scenarios
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task3 = progress.add_task("📋 Parsing scenarios...", total=None)
                scenarios = self._parse_llm_scenarios(llm_response, feature_names, context)
                progress.update(task3, completed=True)
            
            results['scenarios_discovered'] = len(scenarios)
            
            if not scenarios:
                results['errors'].append("No valid scenarios extracted from LLM response")
                return results
            
            # Step 4: Generate comprehensive dataset
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task4 = progress.add_task(f"📊 Generating {output_size:,} row dataset...", total=None)
                dataset, dataset_stats = self._generate_dataset_from_scenarios(
                    scenarios, feature_names, output_size
                )
                progress.update(task4, completed=True)
            
            results['dataset_stats'] = dataset_stats
            
            # Step 5: Export in requested formats
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task5 = progress.add_task("💾 Exporting dataset...", total=None)
                exported_files = self._export_dataset(
                    dataset, feature_names, context, output_formats, output_dir
                )
                progress.update(task5, completed=True)
            
            results['generated_files'] = exported_files
            results['success'] = True
            
            # Display success summary
            self._display_success_summary(results)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            results['errors'].append(str(e))
            self.console.print(f"[red]❌ Error in pipeline: {e}[/red]")
            self.console.print(f"[red]Full traceback:[/red]")
            self.console.print(error_details)
        
        return results
    
    def _create_scenario_discovery_prompt(self, feature_names: List[str], context: str) -> str:
        """Create optimized LLM prompt for scenario discovery"""
        
        features_str = ", ".join(feature_names)
        
        prompt = f"""Reasoning: high

You are an expert in {context} systems and ML model testing. 

TASK: Generate ALL possible testing scenarios for a comprehensive {context} model with these features:
{features_str}

REQUIREMENTS: Create scenarios covering:

1. DEMOGRAPHIC COMBINATIONS (for bias testing):
   - All age groups × education levels × employment types
   - Protected class combinations (gender, race, disability, etc.)
   - Geographic and socioeconomic variations

2. DOMAIN-SPECIFIC PROFILES:
   - Financial ranges: Very low, Low, Medium, High, Very high
   - Risk patterns: No history, Poor, Fair, Good, Excellent
   - Behavioral scenarios: Different usage patterns
   - Status variations: Active, inactive, new, established

3. EDGE CASES & ANOMALIES:
   - Unusual but valid combinations
   - Boundary value scenarios
   - Contradictory pattern cases
   - Rare but important situations

4. BIAS TESTING SCENARIOS:
   - Protected class combinations that might reveal bias
   - Same profile across different demographics
   - Historical discrimination patterns to test against

5. STRESS TESTING:
   - Extreme values within valid ranges
   - Mathematical boundary conditions
   - System limit scenarios

For each scenario, provide:
- Scenario name and description
- Why it's important for testing
- Which features are most relevant
- Expected realistic value patterns
- What model behavior it might reveal

Focus on {context} domain expertise. Be exhaustive - this is for production-level testing where we must catch every possible edge case and bias.

Generate at least 15-20 distinct scenarios that together provide complete test coverage."""

        return prompt
    
    def _save_llm_interaction(self, prompt: str, response: str):
        """Save LLM prompt and response to file for debugging/improvement"""
        
        # Create interactions directory
        interactions_dir = Path("llm_interactions")
        interactions_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save interaction
        interaction_file = interactions_dir / f"interaction_{timestamp}.txt"
        
        with open(interaction_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"LLM INTERACTION - {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.llm_model}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt)
            f.write("\n\n")
            
            f.write("RESPONSE:\n")
            f.write("-" * 40 + "\n")
            f.write(response)
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
        
        self.console.print(f"[dim]💾 Saved LLM interaction to: {interaction_file}[/dim]")
    
    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query LLM using Ollama with the given prompt"""
        
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
                    timeout=600  # 10 minute timeout for M1 MacBook Air
                )
            
            # Cleanup
            os.unlink(prompt_file)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Save prompt and response to file
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
    
    def _parse_llm_scenarios(self, llm_response: str, feature_names: List[str], context: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured scenarios"""
        
        scenarios = []
        
        # Extract scenarios using pattern matching
        # Look for scenario patterns in the response
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
                    scenario = self._create_scenario_template(name, description, feature_names, context)
                    if scenario:
                        scenarios.append(scenario)
        
        # If pattern matching fails, create default scenarios based on context
        if len(scenarios) < 5:
            scenarios.extend(self._create_default_scenarios(feature_names, context))
        
        return scenarios[:20]  # Limit to top 20 scenarios
    
    def _create_scenario_template(self, name: str, description: str, 
                                feature_names: List[str], context: str) -> Optional[Dict[str, Any]]:
        """Create scenario template from name and description"""
        
        scenario = {
            'name': name[:50],  # Limit name length
            'description': description[:200],  # Limit description
            'template': {},
            'priority': 'medium'
        }
        
        # Analyze scenario content to determine priority and constraints
        name_lower = name.lower()
        desc_lower = description.lower()
        
        # High priority scenarios (bias testing, edge cases)
        if any(term in name_lower + desc_lower for term in [
            'bias', 'edge', 'extreme', 'rare', 'unusual', 'protected', 'discrimination',
            'minority', 'lgbtq', 'disability', 'immigrant', 'senior', 'young'
        ]):
            scenario['priority'] = 'critical'
        
        elif any(term in name_lower + desc_lower for term in [
            'high', 'low', 'stress', 'boundary', 'limit', 'maximum', 'minimum'
        ]):
            scenario['priority'] = 'high'
        
        # Extract constraints based on content
        if context.lower() in ['loan', 'credit', 'banking', 'finance']:
            scenario['template'] = self._extract_financial_constraints(name_lower + desc_lower)
        elif context.lower() in ['fraud', 'security', 'risk']:
            scenario['template'] = self._extract_security_constraints(name_lower + desc_lower)
        else:
            scenario['template'] = self._extract_generic_constraints(name_lower + desc_lower, feature_names)
        
        return scenario if scenario['template'] else None
    
    def _extract_financial_constraints(self, content: str) -> Dict[str, Any]:
        """Extract financial scenario constraints"""
        constraints = {}
        
        # Income patterns
        if 'high income' in content or 'wealthy' in content:
            constraints['income'] = (100000, 500000)
        elif 'low income' in content or 'poor' in content:
            constraints['income'] = (15000, 40000)
        elif 'middle income' in content:
            constraints['income'] = (50000, 100000)
        
        # Credit patterns
        if 'excellent credit' in content or 'high credit' in content:
            constraints['credit_score'] = (750, 850)
        elif 'poor credit' in content or 'bad credit' in content:
            constraints['credit_score'] = (300, 580)
        elif 'no credit' in content:
            constraints['credit_score'] = [0]
        
        # Age patterns
        if 'young' in content or 'youth' in content:
            constraints['age'] = (18, 30)
        elif 'senior' in content or 'elderly' in content:
            constraints['age'] = (65, 80)
        elif 'middle' in content:
            constraints['age'] = (35, 55)
        
        # Employment patterns
        if 'unemployed' in content:
            constraints['employment_status'] = ['Unemployed']
        elif 'self-employed' in content or 'entrepreneur' in content:
            constraints['employment_status'] = ['Self-Employed']
        elif 'employed' in content:
            constraints['employment_status'] = ['Employed Full-Time']
        
        return constraints
    
    def _extract_security_constraints(self, content: str) -> Dict[str, Any]:
        """Extract security/fraud scenario constraints"""
        constraints = {}
        
        # Risk patterns
        if 'high risk' in content:
            constraints['risk_score'] = (0.7, 1.0)
        elif 'low risk' in content:
            constraints['risk_score'] = (0.0, 0.3)
        
        # Transaction patterns
        if 'high volume' in content or 'frequent' in content:
            constraints['transaction_frequency'] = (50, 200)
        elif 'low volume' in content or 'rare' in content:
            constraints['transaction_frequency'] = (1, 10)
        
        return constraints
    
    def _extract_generic_constraints(self, content: str, feature_names: List[str]) -> Dict[str, Any]:
        """Extract generic constraints for any domain"""
        constraints = {}
        
        # Look for numeric features and apply basic patterns
        numeric_indicators = ['age', 'income', 'amount', 'score', 'count', 'value', 'price']
        categorical_indicators = ['type', 'status', 'category', 'level', 'class', 'group']
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(indicator in feature_lower for indicator in numeric_indicators):
                if 'high' in content:
                    constraints[feature] = (70, 100)
                elif 'low' in content:
                    constraints[feature] = (0, 30)
                else:
                    constraints[feature] = (20, 80)
        
        return constraints
    
    def _create_default_scenarios(self, feature_names: List[str], context: str) -> List[Dict[str, Any]]:
        """Create default scenarios if LLM parsing fails"""
        
        default_scenarios = [
            {
                'name': 'High Value Profile',
                'description': 'High-value customers with excellent metrics',
                'template': {'income': (80000, 200000), 'age': (30, 50)},
                'priority': 'high'
            },
            {
                'name': 'Low Value Profile',
                'description': 'Low-value customers with poor metrics',
                'template': {'income': (15000, 40000), 'age': (18, 65)},
                'priority': 'high'
            },
            {
                'name': 'Edge Case Profile',
                'description': 'Unusual combinations for edge case testing',
                'template': {'age': (18, 25), 'income': (100000, 300000)},
                'priority': 'critical'
            },
            {
                'name': 'Senior Population',
                'description': 'Senior citizens for age bias testing',
                'template': {'age': (65, 80), 'income': (20000, 60000)},
                'priority': 'critical'
            },
            {
                'name': 'Young Professional',
                'description': 'Young professionals starting careers',
                'template': {'age': (22, 30), 'income': (40000, 80000)},
                'priority': 'medium'
            }
        ]
        
        return default_scenarios
    
    def _generate_dataset_from_scenarios(self, scenarios: List[Dict[str, Any]], 
                                       feature_names: List[str], 
                                       output_size: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate dataset from parsed scenarios"""
        
        # Convert scenarios to LoanScenario objects
        loan_scenarios = []
        for scenario in scenarios:
            loan_scenario = LoanScenario(
                name=scenario['name'],
                description=scenario['description'],
                template=scenario['template'],
                priority=scenario['priority']
            )
            loan_scenarios.append(loan_scenario)
        
        # Initialize generator
        generator = LoanDatasetGenerator(seed=42)
        generator.scenarios = loan_scenarios
        
        # Generate dataset
        dataset = generator.generate_dataset(
            total_rows=output_size, 
            bias_testing_ratio=0.4  # 40% bias testing
        )
        
        # Filter to only requested features (if they exist in dataset)
        available_features = [f for f in feature_names if f in dataset.columns]
        if available_features and len(available_features) == len(feature_names):
            # Only filter if ALL requested features exist
            dataset = dataset[available_features]
        elif available_features:
            # If some features are missing, keep all generated features but warn
            missing_features = set(feature_names) - set(available_features)
            print(f"Note: Some requested features not generated: {missing_features}")
            print(f"Generated dataset includes: {list(dataset.columns)}")
        
        # Generate stats
        stats = generator.generate_data_summary(dataset)
        
        return dataset, stats
    
    def _export_dataset(self, dataset: pd.DataFrame, feature_names: List[str], 
                       context: str, output_formats: List[str], output_dir: str) -> List[str]:
        """Export dataset in requested formats"""
        
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
            elif fmt == 'excel':
                filepath = Path(output_dir) / f"{base_filename}.xlsx"
                dataset.to_excel(filepath, index=False)
            
            exported_files.append(str(filepath))
        
        return exported_files
    
    def _display_success_summary(self, results: Dict[str, Any]):
        """Display generation success summary"""
        
        # Success panel
        success_text = f"""
✅ [green]Dataset Generation Complete![/green]

📊 **Generated Dataset:**
• Rows: {results['dataset_stats'].get('dataset_info', {}).get('total_rows', 'N/A'):,}
• Columns: {len(results['feature_names'])}
• Context: {results['context']}

🧠 **LLM Analysis:**
• Scenarios discovered: {results['scenarios_discovered']}
• Bias testing coverage: Comprehensive

💾 **Output Files:**
{chr(10).join(f"• {Path(f).name}" for f in results['generated_files'])}
        """
        
        self.console.print(Panel(success_text, title="✅ Success", border_style="green"))
        
        # Dataset stats table
        if 'dataset_stats' in results and 'target_distribution' in results['dataset_stats']:
            stats_table = Table(title="Dataset Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            target_dist = results['dataset_stats']['target_distribution']
            for target, count in target_dist.items():
                stats_table.add_row(f"Target: {target}", f"{count:,}")
            
            self.console.print(stats_table)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Automated Dataset Generation CLI - From feature names to comprehensive testing dataset"
    )
    
    parser.add_argument(
        'features',
        nargs='+',
        help='Feature names (space-separated)'
    )
    
    parser.add_argument(
        '--context',
        required=True,
        help='Domain context (e.g., "loan approval", "fraud detection", "customer segmentation")'
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
        choices=['csv', 'json', 'parquet', 'excel'],
        default=['csv'],
        help='Output formats (default: csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='generated_datasets',
        help='Output directory (default: generated_datasets)'
    )
    
    parser.add_argument(
        '--llm-model',
        default='gpt-oss:20b',
        help='LLM model to use (default: gpt-oss:20b)'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = AutoDatasetCLI(llm_model=args.llm_model)
    
    # Display input summary
    console.print(Panel(
        f"🎯 **Input Features:** {', '.join(args.features[:5])}{'...' if len(args.features) > 5 else ''}\n"
        f"🎯 **Context:** {args.context}\n"
        f"🎯 **Target Size:** {args.size:,} rows\n"
        f"🎯 **Formats:** {', '.join(args.formats)}",
        title="🚀 Auto Dataset Generation",
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
        console.print(f"\n[green]🎉 Success! Generated files in: {args.output_dir}[/green]")
        sys.exit(0)
    else:
        console.print(f"\n[red]❌ Generation failed: {', '.join(results['errors'])}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()