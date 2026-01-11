#!/usr/bin/env python3
"""
Quick Dataset Generation CLI - Fast mode without LLM
For M1 MacBook Air and quick testing
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Import our generator
from loan_dataset_generator import LoanDatasetGenerator, LoanScenario

console = Console()

def create_quick_scenarios(feature_names, context):
    """Create predefined scenarios without LLM for fast generation"""
    
    scenarios = []
    
    if context.lower() in ['loan', 'credit', 'banking', 'finance']:
        # Financial scenarios
        scenarios = [
            {
                'name': 'High Income Excellent Credit',
                'description': 'High earners with excellent credit scores',
                'template': {'income': (80000, 200000), 'credit_score': (750, 850), 'age': (30, 55)},
                'priority': 'high'
            },
            {
                'name': 'Low Income Poor Credit',
                'description': 'Low income applicants with poor credit',
                'template': {'income': (15000, 40000), 'credit_score': (300, 580), 'age': (25, 65)},
                'priority': 'high'
            },
            {
                'name': 'Young Professional',
                'description': 'Young professionals starting careers',
                'template': {'age': (22, 30), 'income': (40000, 80000), 'credit_score': (650, 750)},
                'priority': 'medium'
            },
            {
                'name': 'Senior Citizen',
                'description': 'Senior citizens for age bias testing',
                'template': {'age': (65, 80), 'income': (20000, 60000), 'credit_score': (600, 800)},
                'priority': 'critical'
            },
            {
                'name': 'No Credit History',
                'description': 'Applicants with no credit history',
                'template': {'credit_score': [0], 'age': (18, 35), 'income': (25000, 70000)},
                'priority': 'critical'
            },
            {
                'name': 'High Income Low Credit',
                'description': 'High earners with poor credit (edge case)',
                'template': {'income': (100000, 300000), 'credit_score': (300, 600), 'age': (30, 50)},
                'priority': 'critical'
            },
            {
                'name': 'Middle Income Stable',
                'description': 'Stable middle-income households',
                'template': {'income': (50000, 90000), 'credit_score': (680, 780), 'age': (35, 55)},
                'priority': 'medium'
            },
            {
                'name': 'Self Employed',
                'description': 'Self-employed individuals',
                'template': {'employment_status': ['Self-Employed'], 'income': (30000, 150000), 'age': (25, 60)},
                'priority': 'high'
            },
            {
                'name': 'Unemployed Applicant',
                'description': 'Unemployed individuals (bias testing)',
                'template': {'employment_status': ['Unemployed'], 'income': (0, 20000), 'age': (22, 55)},
                'priority': 'critical'
            },
            {
                'name': 'Business Loan Seekers',
                'description': 'Business loan applicants',
                'template': {'loan_type': ['Business', 'Business Expansion'], 'income': (60000, 250000), 'age': (30, 65)},
                'priority': 'medium'
            }
        ]
    
    elif context.lower() in ['fraud', 'security', 'risk']:
        # Security/fraud scenarios
        scenarios = [
            {
                'name': 'High Risk Profile',
                'description': 'High-risk users for fraud testing',
                'template': {'risk_score': (0.7, 1.0), 'age': (20, 40)},
                'priority': 'critical'
            },
            {
                'name': 'Low Risk Profile', 
                'description': 'Low-risk established users',
                'template': {'risk_score': (0.0, 0.3), 'age': (35, 65)},
                'priority': 'medium'
            },
            {
                'name': 'New User Profile',
                'description': 'New users with limited history',
                'template': {'account_age': (0, 30), 'transaction_count': (1, 10)},
                'priority': 'high'
            }
        ]
    
    else:
        # Generic scenarios for any domain
        scenarios = [
            {
                'name': 'High Value Users',
                'description': 'High-value user segment',
                'template': {'value_score': (80, 100), 'age': (30, 55)},
                'priority': 'high'
            },
            {
                'name': 'Low Value Users',
                'description': 'Low-value user segment', 
                'template': {'value_score': (0, 30), 'age': (18, 70)},
                'priority': 'high'
            },
            {
                'name': 'Edge Case Users',
                'description': 'Unusual user patterns',
                'template': {'age': (18, 25), 'value_score': (70, 100)},
                'priority': 'critical'
            },
            {
                'name': 'Senior Users',
                'description': 'Senior user demographic',
                'template': {'age': (65, 80)},
                'priority': 'critical'
            },
            {
                'name': 'Young Users',
                'description': 'Young user demographic',
                'template': {'age': (18, 25)},
                'priority': 'medium'
            }
        ]
    
    return scenarios

def quick_generate_dataset(feature_names, context, output_size, output_formats, output_dir):
    """Quick dataset generation without LLM"""
    
    console.print(Panel(
        f"[bold yellow]Quick Mode - No LLM Required[/bold yellow]\n"
        f"Features: {len(feature_names)} columns\n"
        f"Context: {context}\n"
        f"Target size: {output_size:,} rows",
        title="Fast Generation Mode"
    ))
    
    # Step 1: Create scenarios
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task1 = progress.add_task("Creating predefined scenarios...", total=None)
        scenario_defs = create_quick_scenarios(feature_names, context)
        progress.update(task1, completed=True)
    
    # Step 2: Convert to LoanScenario objects
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task2 = progress.add_task("Setting up scenarios...", total=None)
        
        loan_scenarios = []
        for scenario_def in scenario_defs:
            loan_scenario = LoanScenario(
                name=scenario_def['name'],
                description=scenario_def['description'],
                template=scenario_def['template'],
                priority=scenario_def['priority']
            )
            loan_scenarios.append(loan_scenario)
        
        progress.update(task2, completed=True)
    
    # Step 3: Generate dataset
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task3 = progress.add_task(f"Generating {output_size:,} row dataset...", total=None)
        
        # Initialize generator
        generator = LoanDatasetGenerator(seed=42)
        generator.scenarios = loan_scenarios
        
        # Generate dataset
        dataset = generator.generate_dataset(
            total_rows=output_size, 
            bias_testing_ratio=0.4
        )
        
        # Filter to requested features if they exist
        available_features = [f for f in feature_names if f in dataset.columns]
        if available_features:
            dataset = dataset[available_features]
        
        progress.update(task3, completed=True)
    
    # Step 4: Export dataset
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task4 = progress.add_task("Exporting dataset...", total=None)
        
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
        
        progress.update(task4, completed=True)
    
    # Generate summary stats
    stats = {
        'total_rows': len(dataset),
        'total_columns': len(dataset.columns),
        'scenarios_used': len(loan_scenarios),
        'target_distribution': dataset['target'].value_counts().to_dict() if 'target' in dataset.columns else {}
    }
    
    # Display success summary
    success_text = f"""
[green]Quick Dataset Generation Complete![/green]

**Generated Dataset:**
• Rows: {stats['total_rows']:,}
• Columns: {stats['total_columns']}
• Scenarios: {stats['scenarios_used']}
• Context: {context}

**Output Files:**
{chr(10).join(f"• {Path(f).name}" for f in exported_files)}
    """
    
    console.print(Panel(success_text, title="Success", border_style="green"))
    
    # Show target distribution if available
    if stats['target_distribution']:
        stats_table = Table(title="Target Distribution")
        stats_table.add_column("Target", style="cyan")
        stats_table.add_column("Count", style="white")
        
        for target, count in stats['target_distribution'].items():
            stats_table.add_row(target, f"{count:,}")
        
        console.print(stats_table)
    
    return exported_files

def main():
    """Main CLI interface for quick mode"""
    parser = argparse.ArgumentParser(
        description="Quick Dataset Generation CLI - Fast mode without LLM for M1 MacBook Air"
    )
    
    parser.add_argument(
        'features',
        nargs='+',
        help='Feature names (space-separated)'
    )
    
    parser.add_argument(
        '--context',
        required=True,
        help='Domain context (e.g., "loan approval", "fraud detection")'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=1000,
        help='Dataset size (default: 1000)'
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
        default='quick_datasets',
        help='Output directory (default: quick_datasets)'
    )
    
    args = parser.parse_args()
    
    # Display input summary
    console.print(Panel(
        f"**Features:** {', '.join(args.features[:5])}{'...' if len(args.features) > 5 else ''}\n"
        f"**Context:** {args.context}\n"
        f"**Size:** {args.size:,} rows\n"
        f"**Formats:** {', '.join(args.formats)}",
        title="Quick Dataset Generation",
        border_style="yellow"
    ))
    
    try:
        # Generate dataset quickly
        exported_files = quick_generate_dataset(
            feature_names=args.features,
            context=args.context,
            output_size=args.size,
            output_formats=args.formats,
            output_dir=args.output_dir
        )
        
        console.print(f"\n[green]Success! Generated files in: {args.output_dir}[/green]")
        console.print(f"[cyan]Tip: Use auto_dataset_cli.py for LLM-powered comprehensive scenarios[/cyan]")
        
    except Exception as e:
        console.print(f"\n[red]Generation failed: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
