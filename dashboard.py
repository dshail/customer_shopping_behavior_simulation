
"""
Customer Shopping Behavior Simulation - Interactive Dashboard
===========================================================

Streamlit-based interactive dashboard for exploring simulation results.

Author: dshail
Date: 2025

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')
from analysis import SimulationAnalyzer


# Page configuration
st.set_page_config(
    page_title="Customer Behavior Simulation Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_simulation_data():
    """Load simulation data with caching"""
    try:
        transactions_df = pd.read_csv('data/output/transactions.csv')
        customers_df = pd.read_csv('data/output/customers.csv')

        # Convert date columns
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

        return transactions_df, customers_df
    except FileNotFoundError:
        st.error("Simulation data not found. Please run the simulation first using: python main.py")
        return None, None


def display_header():
    """Display dashboard header"""
    st.title("🛒 Customer Shopping Behavior Simulation Dashboard")
    st.markdown("""
    **Interactive analysis of synthetic customer behavior data generated for retail analytics.**

    This dashboard provides comprehensive insights into customer shopping patterns, persona performance, 
    and temporal variations in purchasing behavior.
    """)


def display_key_metrics(analyzer):
    """Display key performance metrics"""

    exec_summary = analyzer.generate_executive_summary()
    metrics = exec_summary['key_metrics']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Revenue",
            value=f"₹{metrics['total_revenue']:,.0f}",
            delta=None
        )

    with col2:
        st.metric(
            label="Total Transactions", 
            value=f"{metrics['total_transactions']:,}",
            delta=None
        )

    with col3:
        st.metric(
            label="Unique Customers",
            value=f"{metrics['unique_customers']:,}",
            delta=None
        )

    with col4:
        st.metric(
            label="Avg Transaction Value",
            value=f"₹{metrics['average_transaction_value']:,.0f}",
            delta=None
        )


def display_persona_analysis(analyzer, transactions_df):
    """Display persona performance analysis"""

    st.header("👥 Persona Performance Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Revenue by persona
        persona_revenue = transactions_df.groupby('persona_type')['total_amount'].sum().sort_values(ascending=False)

        fig_revenue = px.bar(
            x=persona_revenue.values,
            y=persona_revenue.index,
            orientation='h',
            title="Total Revenue by Customer Persona",
            labels={'x': 'Revenue (₹)', 'y': 'Persona Type'},
            color=persona_revenue.values,
            color_continuous_scale='viridis'
        )
        fig_revenue.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col2:
        # Persona metrics table
        persona_perf = analyzer.analyze_persona_performance()

        st.subheader("Performance Metrics")
        display_df = persona_perf[['total_revenue', 'avg_transaction_value', 'total_transactions']].round(0)
        display_df.columns = ['Revenue (₹)', 'Avg Transaction (₹)', 'Transactions']
        st.dataframe(display_df)

    # Transaction value distribution by persona
    st.subheader("Transaction Value Distribution by Persona")

    fig_dist = px.box(
        transactions_df,
        x='persona_type',
        y='total_amount',
        title="Transaction Value Distribution",
        labels={'total_amount': 'Transaction Amount (₹)', 'persona_type': 'Persona Type'}
    )
    fig_dist.update_xaxes(tickangle=45)
    st.plotly_chart(fig_dist, use_container_width=True)


def display_temporal_analysis(analyzer, transactions_df):
    """Display temporal analysis"""

    st.header("📅 Temporal Shopping Patterns")

    # Daily trends
    daily_trends = transactions_df.groupby('date').agg({
        'total_amount': 'sum',
        'transaction_id': 'count'
    }).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig_daily = px.line(
            daily_trends,
            x='date',
            y='total_amount',
            title="Daily Revenue Trends",
            labels={'total_amount': 'Daily Revenue (₹)', 'date': 'Date'}
        )
        fig_daily.update_traces(line_color='#1f77b4', line_width=2)
        st.plotly_chart(fig_daily, use_container_width=True)

    with col2:
        fig_transactions = px.line(
            daily_trends,
            x='date', 
            y='transaction_id',
            title="Daily Transaction Count",
            labels={'transaction_id': 'Number of Transactions', 'date': 'Date'}
        )
        fig_transactions.update_traces(line_color='#ff7f0e', line_width=2)
        st.plotly_chart(fig_transactions, use_container_width=True)

    # Day of week analysis
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    transactions_df['day_of_week'] = pd.Categorical(
        transactions_df['timestamp'].dt.day_name(), 
        categories=dow_order, 
        ordered=True
    )

    dow_analysis = transactions_df.groupby('day_of_week').agg({
        'total_amount': ['sum', 'mean'],
        'transaction_id': 'count'
    }).round(2)

    col1, col2 = st.columns(2)

    with col1:
        fig_dow = px.bar(
            x=dow_analysis.index,
            y=dow_analysis[('total_amount', 'sum')],
            title="Total Revenue by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Total Revenue (₹)'}
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col2:
        # Hourly patterns
        hourly_patterns = transactions_df.groupby(transactions_df['timestamp'].dt.hour)['transaction_id'].count()

        fig_hourly = px.bar(
            x=hourly_patterns.index,
            y=hourly_patterns.values,
            title="Transaction Count by Hour of Day",
            labels={'x': 'Hour of Day', 'y': 'Number of Transactions'}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    # Festival vs Regular comparison
    st.subheader("Festival vs Regular Day Analysis")

    festival_comparison = transactions_df.groupby('is_festival').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'num_items': 'mean'
    }).round(2)

    col1, col2 = st.columns(2)

    with col1:
        festival_labels = ['Regular Days', 'Festival Periods']
        festival_revenue = [
            festival_comparison.loc[False, ('total_amount', 'sum')] if False in festival_comparison.index else 0,
            festival_comparison.loc[True, ('total_amount', 'sum')] if True in festival_comparison.index else 0
        ]

        fig_festival = px.bar(
            x=festival_labels,
            y=festival_revenue,
            title="Total Revenue: Festival vs Regular Days",
            labels={'x': 'Day Type', 'y': 'Total Revenue (₹)'}
        )
        st.plotly_chart(fig_festival, use_container_width=True)

    with col2:
        festival_avg = [
            festival_comparison.loc[False, ('total_amount', 'mean')] if False in festival_comparison.index else 0,
            festival_comparison.loc[True, ('total_amount', 'mean')] if True in festival_comparison.index else 0
        ]

        fig_festival_avg = px.bar(
            x=festival_labels,
            y=festival_avg,
            title="Average Transaction Value: Festival vs Regular",
            labels={'x': 'Day Type', 'y': 'Average Transaction Value (₹)'}
        )
        st.plotly_chart(fig_festival_avg, use_container_width=True)


def display_customer_insights(analyzer, transactions_df, customers_df):
    """Display customer segmentation insights"""

    st.header("🎯 Customer Insights & Segmentation")

    # Customer lifetime value distribution
    customer_ltv = transactions_df.groupby('customer_id')['total_amount'].sum()

    col1, col2 = st.columns(2)

    with col1:
        fig_ltv = px.histogram(
            x=customer_ltv.values,
            nbins=50,
            title="Customer Lifetime Value Distribution",
            labels={'x': 'Customer Lifetime Value (₹)', 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig_ltv, use_container_width=True)

    with col2:
        # Shopping frequency distribution
        customer_frequency = transactions_df.groupby('customer_id').size()

        fig_freq = px.histogram(
            x=customer_frequency.values,
            nbins=20,
            title="Customer Shopping Frequency Distribution", 
            labels={'x': 'Number of Transactions', 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    # Top customers analysis
    st.subheader("Top Performing Customers")

    top_customers = customer_ltv.nlargest(10).reset_index()
    top_customers_with_info = top_customers.merge(
        customers_df[['customer_id', 'name', 'persona_type', 'age', 'income']],
        on='customer_id'
    )

    display_top_customers = top_customers_with_info[
        ['name', 'persona_type', 'age', 'income', 'total_amount']
    ].round(2)
    display_top_customers.columns = ['Customer Name', 'Persona', 'Age', 'Income (₹)', 'Total Spent (₹)']

    st.dataframe(display_top_customers, use_container_width=True)


def display_data_quality_report(analyzer):
    """Display data quality assessment"""

    with st.expander("📊 Data Quality Report", expanded=False):
        quality_report = analyzer.validate_data_quality()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Completeness")
            st.json(quality_report['data_completeness'])

            st.subheader("Data Consistency")
            st.json(quality_report['data_consistency'])

        with col2:
            st.subheader("Outlier Analysis")
            st.json(quality_report['outlier_analysis'])

            st.subheader("Business Logic Validation")
            st.json(quality_report['business_logic_validation'])


def display_simulation_config():
    """Display simulation configuration"""

    with st.expander("⚙️ Simulation Configuration", expanded=False):
        try:
            with open('data/output/simulation_config.json', 'r') as f:
                config = json.load(f)
            st.json(config)
        except FileNotFoundError:
            st.warning("Simulation configuration file not found.")


def main():
    """Main dashboard application"""

    display_header()

    # Load data
    transactions_df, customers_df = load_simulation_data()

    if transactions_df is None or customers_df is None:
        st.stop()

    # Initialize analyzer
    analyzer = SimulationAnalyzer(transactions_df, customers_df)

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Date range filter
    min_date = transactions_df['date'].min().date()
    max_date = transactions_df['date'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Persona filter
    personas = ['All'] + list(transactions_df['persona_type'].unique())
    selected_personas = st.sidebar.multiselect(
        "Select Personas",
        personas,
        default=['All']
    )

    # Apply filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        transactions_df = transactions_df[
            (transactions_df['date'].dt.date >= start_date) & 
            (transactions_df['date'].dt.date <= end_date)
        ]

    if 'All' not in selected_personas and selected_personas:
        transactions_df = transactions_df[transactions_df['persona_type'].isin(selected_personas)]

    # Update analyzer with filtered data
    analyzer = SimulationAnalyzer(transactions_df, customers_df)

    # Display sections
    display_key_metrics(analyzer)

    st.divider()
    display_persona_analysis(analyzer, transactions_df)

    st.divider()
    display_temporal_analysis(analyzer, transactions_df)

    st.divider()
    display_customer_insights(analyzer, transactions_df, customers_df)

    st.divider()
    display_data_quality_report(analyzer)
    display_simulation_config()

    # Footer
    st.markdown("---")
    st.markdown("""
    **Customer Shopping Behavior Simulation Dashboard**  
    Built with <3  
    Author: dshail | Date: August 2025
    """)


if __name__ == "__main__":
    main()
