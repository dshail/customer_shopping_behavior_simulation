
"""
Customer Shopping Behavior Simulation - Main Application
======================================================

Main entry point for running the customer shopping behavior simulation.

Author: dshail
Date: 2025

Usage:
    python main.py --days 30 --customers 1000
    python main.py --config config/custom_personas.yaml
    python main.py --help
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from simulator import CustomerBehaviorSimulator
import logging


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup application logging"""

    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_banner() -> None:
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🛒  CUSTOMER SHOPPING BEHAVIOR SIMULATION SYSTEM              ║
║                                                                  ║
║   Built for: Stimulation                                         ║
║   Author: dshail                                                 ║
║   Date: August 2025                                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_simulation_summary(summary: dict) -> None:
    """Print formatted simulation summary"""

    print("\n" + "="*70)
    print("📊 SIMULATION RESULTS SUMMARY")
    print("="*70)

    # Overview
    overview = summary['overview']
    print(f"\n🔍 OVERVIEW:")
    print(f"   Total Customers: {overview['total_customers']:,}")
    print(f"   Total Transactions: {overview['total_transactions']:,}")
    print(f"   Simulation Period: {overview['simulation_period']}")
    print(f"   Total Revenue: ₹{overview['total_revenue']:,.2f}")
    print(f"   Average Transaction Value: ₹{overview['avg_transaction_value']:,.2f}")

    # Persona breakdown
    print(f"\n👥 PERSONA PERFORMANCE:")
    for persona_name, stats in summary['persona_breakdown'].items():
        print(f"   {persona_name}:")
        print(f"      Revenue: ₹{stats['total_revenue']:,.2f}")
        print(f"      Transactions: {stats['transaction_count']:,}")
        print(f"      Avg Transaction: ₹{stats['avg_transaction']:,.2f}")
        print(f"      Avg Items/Basket: {stats['avg_items_per_basket']:.1f}")

    # Temporal analysis
    temporal = summary['temporal_analysis']
    print(f"\n📅 TEMPORAL ANALYSIS:")

    festival_data = temporal['festival_vs_regular']
    print(f"   Festival vs Regular Days:")
    print(f"      Festival Revenue: ₹{festival_data['festival_revenue']:,.2f}")
    print(f"      Regular Revenue: ₹{festival_data['regular_revenue']:,.2f}")
    print(f"      Festival Avg: ₹{festival_data['festival_avg_transaction']:,.2f}")
    print(f"      Regular Avg: ₹{festival_data['regular_avg_transaction']:,.2f}")

    weekend_data = temporal['weekend_vs_weekday']
    print(f"   Weekend vs Weekday:")
    print(f"      Weekend Revenue: ₹{weekend_data['weekend_revenue']:,.2f}")
    print(f"      Weekday Revenue: ₹{weekend_data['weekday_revenue']:,.2f}")
    print(f"      Weekend Avg: ₹{weekend_data['weekend_avg_transaction']:,.2f}")
    print(f"      Weekday Avg: ₹{weekend_data['weekday_avg_transaction']:,.2f}")

    # Performance metrics
    perf = summary['performance_metrics']
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Execution Time: {perf['execution_time_seconds']:.2f} seconds")
    print(f"   Transactions/Second: {perf['transactions_per_second']:.2f}")
    print(f"   Errors Encountered: {perf['errors_encountered']}")

    print("\n" + "="*70)


def validate_arguments(args) -> None:
    """Validate command line arguments"""

    if args.days <= 0:
        raise ValueError("Simulation days must be positive")

    if args.customers <= 0:
        raise ValueError("Customers per persona must be positive")

    if not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")


def main() -> None:
    """Main application entry point"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Customer Shopping Behavior Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default settings
  python main.py --days 60 --customers 2000  # Custom simulation parameters
  python main.py --config custom.yaml     # Use custom configuration
  python main.py --output results/        # Custom output directory
  python main.py --log-level DEBUG        # Enable debug logging
        """
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to simulate (default: 30)'
    )

    parser.add_argument(
        '--customers',
        type=int,
        default=1000,
        help='Number of customers per persona (default: 1000)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/personas.yaml',
        help='Path to persona configuration file (default: config/personas.yaml)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/output',
        help='Output directory for results (default: data/output)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--export-summary',
        action='store_true',
        help='Export detailed summary to JSON file'
    )

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logging(args.log_level)

        # Print banner
        print_banner()

        # Validate arguments
        validate_arguments(args)

        print(f"🔧 Configuration:")
        print(f"   Config File: {args.config}")
        print(f"   Simulation Days: {args.days}")
        print(f"   Customers per Persona: {args.customers}")
        print(f"   Random Seed: {args.seed}")
        print(f"   Output Directory: {args.output}")
        print(f"   Log Level: {args.log_level}")

        # Initialize and run simulation
        print(f"\n🚀 Initializing simulation...")
        simulator = CustomerBehaviorSimulator(
            config_path=args.config,
            random_seed=args.seed
        )

        # Override customers per persona if specified
        if args.customers != 1000:
            simulator.config.setdefault('simulation', {})['customers_per_persona'] = args.customers

        # Run simulation
        print(f"\n⏳ Running {args.days}-day simulation...")
        transactions_df, customers_df = simulator.run_simulation(simulation_days=args.days)

        # Generate summary
        summary = simulator.get_simulation_summary(transactions_df, customers_df)

        # Print results
        print_simulation_summary(summary)

        # Export data
        print(f"\n💾 Exporting data to {args.output}/...")
        simulator.export_data(transactions_df, customers_df, args.output)

        # Export summary if requested
        if args.export_summary:
            summary_file = f"{args.output}/detailed_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"   📄 Detailed summary exported to {summary_file}")

        print(f"\n✅ Simulation completed successfully!")
        print(f"📁 Output files:")
        print(f"   📊 transactions.csv - {len(transactions_df):,} transaction records")
        print(f"   👥 customers.csv - {len(customers_df):,} customer profiles")
        print(f"   📈 simulation_metrics.json - Performance metrics")
        print(f"   ⚙️  simulation_config.json - Configuration used")

        if args.export_summary:
            print(f"   📋 detailed_summary.json - Comprehensive analysis")

        print(f"\n🎯 Ready for analysis and presentation!")

    except KeyboardInterrupt:
        print(f"\n⚠️  Simulation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        print(f"\n❌ Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
