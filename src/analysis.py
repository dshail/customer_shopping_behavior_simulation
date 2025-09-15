
"""
Customer Shopping Behavior Simulation - Analysis Utilities
=========================================================

Utility functions for analyzing simulation results and generating insights.

Author: dshail 
Date: August 2025
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SimulationAnalyzer:
    """
    Comprehensive analysis toolkit for customer behavior simulation results.

    Provides methods for:
    - Statistical analysis of customer behavior
    - Revenue and profitability analysis  
    - Temporal pattern analysis
    - Customer segmentation insights
    - Data quality validation
    """

    def __init__(self, transactions_df: pd.DataFrame, customers_df: pd.DataFrame):
        """
        Initialize analyzer with simulation data.

        Args:
            transactions_df: Transaction data
            customers_df: Customer profile data
        """
        self.transactions_df = transactions_df.copy()
        self.customers_df = customers_df.copy()

        # Preprocessing
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """Preprocess data for analysis"""

        # Convert date columns
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
        self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])

        # Add derived columns
        self.transactions_df['day_of_week'] = self.transactions_df['date'].dt.day_name()
        self.transactions_df['week_number'] = self.transactions_df['date'].dt.isocalendar().week
        self.transactions_df['hour'] = self.transactions_df['timestamp'].dt.hour

        # Add customer segmentation based on spending
        customer_spending = self.transactions_df.groupby('customer_id')['total_amount'].sum()
        spending_quartiles = customer_spending.quantile([0.25, 0.5, 0.75])

        def categorize_customer(customer_id):
            spending = customer_spending.get(customer_id, 0)
            if spending <= spending_quartiles[0.25]:
                return 'Low Spender'
            elif spending <= spending_quartiles[0.5]:
                return 'Medium Spender'
            elif spending <= spending_quartiles[0.75]:
                return 'High Spender'
            else:
                return 'Premium Spender'

        self.transactions_df['spending_category'] = self.transactions_df['customer_id'].apply(categorize_customer)

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of simulation results"""

        total_revenue = self.transactions_df['total_amount'].sum()
        total_transactions = len(self.transactions_df)
        unique_customers = self.transactions_df['customer_id'].nunique()
        avg_transaction_value = self.transactions_df['total_amount'].mean()

        # Customer lifetime value
        customer_ltv = self.transactions_df.groupby('customer_id')['total_amount'].sum()

        # Top performing metrics
        top_persona = self.transactions_df.groupby('persona_type')['total_amount'].sum().idxmax()
        peak_day = self.transactions_df.groupby('date')['total_amount'].sum().idxmax()

        return {
            'key_metrics': {
                'total_revenue': round(total_revenue, 2),
                'total_transactions': total_transactions,
                'unique_customers': unique_customers,
                'average_transaction_value': round(avg_transaction_value, 2),
                'average_customer_ltv': round(customer_ltv.mean(), 2),
                'median_customer_ltv': round(customer_ltv.median(), 2)
            },
            'top_performers': {
                'highest_revenue_persona': top_persona,
                'peak_revenue_day': str(peak_day.date()),
                'peak_revenue_amount': round(self.transactions_df[self.transactions_df['date'] == peak_day]['total_amount'].sum(), 2)
            },
            'customer_insights': {
                'repeat_customer_rate': round((customer_ltv > customer_ltv.median()).mean() * 100, 1),
                'high_value_customer_percentage': round((customer_ltv > customer_ltv.quantile(0.8)).mean() * 100, 1)
            }
        }

    def analyze_persona_performance(self) -> pd.DataFrame:
        """Analyze performance metrics by customer persona"""

        persona_analysis = self.transactions_df.groupby('persona_type').agg({
            'total_amount': ['sum', 'mean', 'std', 'count'],
            'num_items': ['mean', 'std'],
            'customer_id': 'nunique'
        }).round(2)

        # Flatten column names
        persona_analysis.columns = [
            'total_revenue', 'avg_transaction_value', 'transaction_value_std', 'total_transactions',
            'avg_items_per_transaction', 'items_per_transaction_std', 'unique_customers'
        ]

        # Calculate additional metrics
        persona_analysis['revenue_per_customer'] = (
            persona_analysis['total_revenue'] / persona_analysis['unique_customers']
        ).round(2)

        persona_analysis['transactions_per_customer'] = (
            persona_analysis['total_transactions'] / persona_analysis['unique_customers']
        ).round(2)

        # Calculate revenue share
        persona_analysis['revenue_share_percent'] = (
            persona_analysis['total_revenue'] / persona_analysis['total_revenue'].sum() * 100
        ).round(1)

        return persona_analysis.sort_values('total_revenue', ascending=False)

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal shopping patterns"""

        # Daily analysis
        daily_analysis = self.transactions_df.groupby('date').agg({
            'total_amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        })
        daily_analysis.columns = ['daily_revenue', 'daily_transactions', 'daily_unique_customers']

        # Day of week analysis
        dow_analysis = self.transactions_df.groupby('day_of_week').agg({
            'total_amount': ['sum', 'mean'],
            'transaction_id': 'count'
        })
        dow_analysis.columns = ['total_revenue', 'avg_transaction_value', 'transaction_count']

        # Hour of day analysis
        hourly_analysis = self.transactions_df.groupby('hour').agg({
            'total_amount': ['sum', 'mean'],
            'transaction_id': 'count'
        })
        hourly_analysis.columns = ['total_revenue', 'avg_transaction_value', 'transaction_count']

        # Festival vs regular analysis
        festival_comparison = self.transactions_df.groupby('is_festival').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'num_items': 'mean'
        })

        # Weekend vs weekday analysis
        weekend_comparison = self.transactions_df.groupby('is_weekend').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'num_items': 'mean'
        })

        return {
            'daily_trends': daily_analysis,
            'day_of_week_patterns': dow_analysis,
            'hourly_patterns': hourly_analysis,
            'festival_impact': festival_comparison,
            'weekend_effect': weekend_comparison
        }

    def analyze_customer_segments(self) -> Dict[str, Any]:
        """Analyze customer segmentation based on behavior"""

        # RFM Analysis (Recency, Frequency, Monetary)
        current_date = self.transactions_df['date'].max()

        rfm_analysis = self.transactions_df.groupby('customer_id').agg({
            'date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'total_amount': 'sum'  # Monetary
        })
        rfm_analysis.columns = ['recency', 'frequency', 'monetary']

        # Add quintile scores
        rfm_analysis['r_score'] = pd.qcut(rfm_analysis['recency'], 5, labels=[5,4,3,2,1])
        rfm_analysis['f_score'] = pd.qcut(rfm_analysis['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_analysis['m_score'] = pd.qcut(rfm_analysis['monetary'], 5, labels=[1,2,3,4,5])

        # Create RFM segments
        def segment_customers(row):
            if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4:
                return 'Champions'
            elif row['r_score'] >= 3 and row['f_score'] >= 3 and row['m_score'] >= 3:
                return 'Loyal Customers'
            elif row['r_score'] >= 4 and row['f_score'] <= 2:
                return 'New Customers'
            elif row['r_score'] <= 2 and row['f_score'] >= 3:
                return 'At Risk'
            else:
                return 'Others'

        rfm_analysis['segment'] = rfm_analysis.apply(segment_customers, axis=1)

        # Spending category analysis
        spending_analysis = self.transactions_df.groupby('spending_category').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        })

        return {
            'rfm_analysis': rfm_analysis,
            'spending_segments': spending_analysis,
            'segment_distribution': rfm_analysis['segment'].value_counts()
        }

    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and identify potential issues"""

        quality_report = {
            'data_completeness': {},
            'data_consistency': {},
            'outlier_analysis': {},
            'business_logic_validation': {}
        }

        # Data completeness
        quality_report['data_completeness'] = {
            'transactions_null_values': self.transactions_df.isnull().sum().to_dict(),
            'customers_null_values': self.customers_df.isnull().sum().to_dict(),
            'transactions_total_rows': len(self.transactions_df),
            'customers_total_rows': len(self.customers_df)
        }

        # Data consistency checks
        customer_ids_transactions = set(self.transactions_df['customer_id'])
        customer_ids_customers = set(self.customers_df['customer_id'])

        quality_report['data_consistency'] = {
            'customer_id_mismatch': len(customer_ids_transactions - customer_ids_customers),
            'duplicate_transactions': self.transactions_df.duplicated().sum(),
            'duplicate_customers': self.customers_df.duplicated(subset=['customer_id']).sum()
        }

        # Outlier analysis
        Q1 = self.transactions_df['total_amount'].quantile(0.25)
        Q3 = self.transactions_df['total_amount'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

        quality_report['outlier_analysis'] = {
            'high_value_outliers': (self.transactions_df['total_amount'] > outlier_threshold).sum(),
            'zero_amount_transactions': (self.transactions_df['total_amount'] <= 0).sum(),
            'zero_items_transactions': (self.transactions_df['num_items'] <= 0).sum()
        }

        # Business logic validation
        quality_report['business_logic_validation'] = {
            'negative_amounts': (self.transactions_df['total_amount'] < 0).sum(),
            'future_dated_transactions': (self.transactions_df['date'] > datetime.now()).sum(),
            'unrealistic_basket_sizes': (self.transactions_df['num_items'] > 50).sum()
        }

        return quality_report

    def generate_insights_report(self) -> str:
        """Generate natural language insights report"""

        persona_perf = self.analyze_persona_performance()
        temporal_patterns = self.analyze_temporal_patterns()
        exec_summary = self.generate_executive_summary()

        # Generate insights
        insights = []

        # Top performing persona
        top_persona = persona_perf.index[0]
        top_persona_revenue = persona_perf.iloc[0]['total_revenue']
        total_revenue = exec_summary['key_metrics']['total_revenue']

        insights.append(f"💎 **{top_persona}** is the highest revenue-generating persona, contributing ₹{top_persona_revenue:,.2f} ({(top_persona_revenue/total_revenue*100):.1f}% of total revenue).")

        # Festival impact
        if 'festival_impact' in temporal_patterns:
            festival_data = temporal_patterns['festival_impact']
            if len(festival_data) > 1:
                festival_avg = festival_data.loc[True, ('total_amount', 'mean')] if True in festival_data.index else 0
                regular_avg = festival_data.loc[False, ('total_amount', 'mean')] if False in festival_data.index else 0

                if festival_avg > regular_avg:
                    increase = ((festival_avg - regular_avg) / regular_avg * 100)
                    insights.append(f"🎉 **Festival periods** show {increase:.1f}% higher average transaction values compared to regular days (₹{festival_avg:.2f} vs ₹{regular_avg:.2f}).")

        # Customer loyalty
        ltv_median = exec_summary['key_metrics']['median_customer_ltv']
        insights.append(f"🎯 **Customer Lifetime Value** median is ₹{ltv_median:,.2f}, indicating strong customer engagement across personas.")

        # Peak shopping times
        hourly_data = temporal_patterns['hourly_patterns']
        peak_hour = hourly_data['transaction_count'].idxmax()
        peak_transactions = hourly_data.loc[peak_hour, 'transaction_count']

        insights.append(f"⏰ **Peak shopping hour** is {peak_hour}:00 with {peak_transactions} transactions, indicating clear customer preference patterns.")

        # Combine insights into report
        report = f"""
# 📊 Customer Shopping Behavior Simulation - Insights Report

## Key Findings

{chr(10).join(f"- {insight}" for insight in insights)}

## Recommendations

- **Focus on {top_persona}**: This persona drives the majority of revenue and should be prioritized in marketing efforts.
- **Capitalize on Festival Periods**: Implement targeted promotions during festival seasons to maximize the observed spending increase.
- **Optimize Store Hours**: Consider staffing adjustments around peak shopping hours ({peak_hour}:00) for better customer service.
- **Customer Retention**: With a median LTV of ₹{ltv_median:,.2f}, focus on programs to increase customer frequency and basket size.

## Data Quality

The simulation generated high-quality, realistic customer behavior data suitable for:
- Machine learning model training
- Business intelligence and analytics
- A/B testing frameworks
- Customer segmentation strategies
"""

        return report


def create_summary_dashboard(transactions_df: pd.DataFrame, customers_df: pd.DataFrame, 
                           output_path: str = 'data/output/dashboard.html') -> str:
    """Create interactive summary dashboard"""

    analyzer = SimulationAnalyzer(transactions_df, customers_df)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Revenue by Persona', 
            'Daily Transaction Trends',
            'Hourly Shopping Patterns',
            'Festival vs Regular Day Impact'
        ],
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # Revenue by persona
    persona_revenue = transactions_df.groupby('persona_type')['total_amount'].sum().sort_values(ascending=True)
    fig.add_trace(
        go.Bar(x=persona_revenue.values, y=persona_revenue.index, orientation='h', name='Revenue'),
        row=1, col=1
    )

    # Daily trends
    daily_trends = transactions_df.groupby('date').agg({
        'total_amount': 'sum',
        'transaction_id': 'count'
    })
    fig.add_trace(
        go.Scatter(x=daily_trends.index, y=daily_trends['total_amount'], mode='lines+markers', name='Daily Revenue'),
        row=1, col=2
    )

    # Hourly patterns
    hourly_patterns = transactions_df.groupby(transactions_df['timestamp'].dt.hour)['transaction_id'].count()
    fig.add_trace(
        go.Bar(x=hourly_patterns.index, y=hourly_patterns.values, name='Hourly Transactions'),
        row=2, col=1
    )

    # Festival impact
    festival_comparison = transactions_df.groupby('is_festival')['total_amount'].mean()
    festival_labels = ['Regular Days', 'Festival Periods']
    fig.add_trace(
        go.Bar(x=festival_labels, y=festival_comparison.values, name='Avg Transaction Value'),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Customer Shopping Behavior Simulation Dashboard",
        showlegend=False
    )

    # Save dashboard
    fig.write_html(output_path)

    return output_path


def export_analysis_report(transactions_df: pd.DataFrame, customers_df: pd.DataFrame, 
                         output_dir: str = 'data/output') -> None:
    """Export comprehensive analysis report"""

    analyzer = SimulationAnalyzer(transactions_df, customers_df)

    # Generate all analyses
    exec_summary = analyzer.generate_executive_summary()
    persona_performance = analyzer.analyze_persona_performance()
    temporal_patterns = analyzer.analyze_temporal_patterns()
    customer_segments = analyzer.analyze_customer_segments()
    data_quality = analyzer.validate_data_quality()
    insights_report = analyzer.generate_insights_report()

    # Export to files
    with open(f'{output_dir}/executive_summary.json', 'w') as f:
        json.dump(exec_summary, f, indent=2, default=str)

    persona_performance.to_csv(f'{output_dir}/persona_performance.csv')

    with open(f'{output_dir}/temporal_analysis.json', 'w') as f:
        json.dump(temporal_patterns, f, indent=2, default=str)

    with open(f'{output_dir}/data_quality_report.json', 'w') as f:
        json.dump(data_quality, f, indent=2, default=str)

    with open(f'{output_dir}/insights_report.md', 'w') as f:
        f.write(insights_report)

    # Create dashboard
    dashboard_path = create_summary_dashboard(transactions_df, customers_df, f'{output_dir}/dashboard.html')

    print(f"📊 Analysis reports exported to {output_dir}/")
    print(f"   📈 dashboard.html - Interactive visualization dashboard")
    print(f"   📋 insights_report.md - Natural language insights")
    print(f"   📊 persona_performance.csv - Detailed persona metrics")
    print(f"   📈 executive_summary.json - Key performance indicators")
