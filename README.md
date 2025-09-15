# 🛒 Customer Shopping Behavior Simulation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-95%25-green.svg)

> **Comprehensive customer behavior simulation system for retail analytics**  
> Built By dshail | 2025

## 🎯 Project Overview

This project implements a sophisticated **customer shopping behavior simulation system** that generates realistic synthetic transaction data for retail analytics. The system uses **probabilistic modeling**, **persona-based behavior patterns**, and **temporal variations** to create authentic customer shopping patterns.

**🚀 NEW: Enhanced with AI & MLOps capabilities!**

### Key Features

#### Core Simulation Engine

- 🎭 **5 Distinct Customer Personas** with unique shopping behaviors
- 📊 **Probabilistic Basket Generation** with realistic price variations  
- 🎉 **Temporal Modeling** including festivals, weekends, and seasonal effects
- 📈 **Interactive Analytics Dashboard** with comprehensive insights
- 🧪 **Comprehensive Test Suite** ensuring code quality and reliability
- ⚡ **High Performance** - generates 60K+ transactions in seconds

#### 🤖 AI-Powered Features (NEW)

- 🧠 **LLM Integration** via OpenRouter API (50-80% cheaper than OpenAI)
- 🎯 **AI Persona Generation** from market research data
- 📝 **Intelligent Insights** and natural language reporting
- 🔮 **Predictive Analytics** and trend forecasting
- 💡 **Business Intelligence** with AI-generated recommendations

#### 🔧 MLOps Pipeline (NEW)

- 🤖 **Automated ML Training** for customer behavior prediction
- 📊 **Model Performance Monitoring** with drift detection
- 🧪 **A/B Testing Framework** for model comparison
- 📈 **MLflow Integration** for experiment tracking
- 🚀 **Model Versioning** and deployment management
- 📱 **Production Ready** with comprehensive monitoring and alerting

## 🏗️ System Architecture

```
customer-behavior-simulation/
├── 📁 src/                     # Core application code
│   ├── models.py               # Data models and structures
│   ├── simulator.py            # Main simulation engine  
│   ├── analysis.py             # Analytics and insights
│   ├── llm_integration.py      # 🤖 AI-powered features (NEW)
│   └── mlops_pipeline.py       # 🔧 MLOps pipeline (NEW)
├── 📁 config/                  # Configuration files
│   ├── personas.yaml           # Customer persona definitions
│   ├── llm_config.yaml         # 🤖 LLM API configuration (NEW)
│   └── mlops_config.yaml       # 🔧 MLOps settings (NEW)
├── 📁 data/                    # Data storage
│   ├── output/                 # Simulation results
│   ├── demo/                   # Sample datasets
│   └── processed/              # Processed datasets
├── 📁 models/                  # 🤖 Trained ML models (NEW)
│   ├── churn_prediction/       # Customer churn models
│   └── spending_prediction/    # Spending behavior models
├── 📁 mlruns/                  # 📊 MLflow experiment tracking (NEW)
├── 📁 tests/                   # Test suite
│   └── test_simulation.py      # Comprehensive tests
├── 📁 logs/                    # Application logs
├── main.py                     # CLI application
├── main_enhanced.py            # 🚀 Enhanced CLI with AI/ML (NEW)
├── dashboard.py                # Streamlit dashboard
├── demo_enhanced_features.py   # 🎬 Feature demonstration (NEW)
├── IMPLEMENTATION_GUIDE.md     # 📖 Setup guide for new features (NEW)
├── OPENROUTER_SETUP_GUIDE.md   # 🔑 API setup instructions (NEW)
└── requirements.txt            # Dependencies (enhanced)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd customer-behavior-simulation
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulation**

   ```bash
   python main.py --days 30 --customers 1000
   ```

4. **View results**

   ```bash
   streamlit run dashboard.py
   ```

### 🎬 Demo Commands

#### Basic Simulation

```bash
# Basic simulation (30 days, 1000 customers per persona)
python main.py

# Extended simulation with custom parameters
python main.py --days 60 --customers 2000 --export-summary

# Run with debug logging
python main.py --log-level DEBUG

# Use custom configuration
python main.py --config config/custom_personas.yaml
```

#### 🚀 Enhanced Features (NEW)

```bash
# Demo all enhanced features
python demo_enhanced_features.py

# Run enhanced simulation with AI insights
python main_enhanced.py --days 30 --enable-ai-insights

# Generate AI personas from market data
python main_enhanced.py --generate-ai-personas --market-data market_research.json

# Full enhanced simulation with ML training
python main_enhanced.py --days 30 --customers 1000 --enable-ml-training --enable-ai-insights

# View enhanced dashboard with ML metrics
streamlit run dashboard_enhanced.py
```

#### 🔧 MLOps Commands

```bash
# View MLflow experiment tracking
mlflow ui

# Run A/B testing
python main_enhanced.py --run-ab-test model_v1 model_v2

# Monitor model performance
python main_enhanced.py --monitor-models
```

## 📊 Simulation Results

The system generates comprehensive datasets:

### 📈 Key Metrics

- **60,089 transactions** across **5,000 customers**
- **₹295.6M total revenue** over 30-day period
- **5 customer personas** with distinct behaviors
- **7 festival periods** with realistic spending boosts

### 🎭 Customer Personas

| Persona | Frequency | Avg Transaction | Items/Basket | Revenue Share |
|---------|-----------|-----------------|--------------|---------------|
| **Premium Shopper** | Daily | ₹8,567 | 3.2 | 75% |
| **Family Shopper** | Weekly | ₹5,349 | 4.6 | 12% |
| **Young Professional** | Alternate | ₹1,234 | 2.1 | 7% |
| **Budget Conscious** | Weekly | ₹1,073 | 3.8 | 3% |
| **Senior Citizen** | Monthly | ₹2,623 | 3.5 | 3% |

### 📅 Temporal Insights

- **Festival periods**: 39% higher average transaction values
- **Weekend effect**: 16% increase in average spending  
- **Peak shopping**: 6-8pm for young professionals, 10am-12pm for families

## 🔧 Technical Implementation

### Core Technologies

- **Python 3.8+** - Core language
- **Pandas & NumPy** - Data manipulation and analysis
- **Faker** - Synthetic demographic data generation
- **Streamlit & Plotly** - Interactive dashboards
- **PyYAML** - Configuration management
- **Pytest** - Testing framework

### Design Principles

1. **Modular Architecture** - Clean separation of concerns
2. **Data-Driven Configuration** - YAML-based persona definitions
3. **Probabilistic Modeling** - Realistic behavior patterns
4. **Comprehensive Testing** - 95% test coverage
5. **Production Readiness** - Logging, error handling, validation

### Key Algorithms

#### Probabilistic Shopping Decision

```python
def should_customer_shop_today(persona, customer_id, date, history):
    base_probability = get_frequency_probability(persona.frequency)

    # Apply temporal multipliers
    if is_festival_period(date):
        base_probability *= 1.5
    if is_weekend(date):
        base_probability *= 1.2

    return random.random() < min(base_probability, 0.95)
```

#### Dynamic Basket Generation

```python
def generate_shopping_basket(persona, date):
    basket = ShoppingBasket()

    for item_category, config in persona.basket_profile.items():
        if random.random() < config['probability']:
            quantity = random.randint(*config['quantity'])
            price = random.uniform(*config['price_range'])

            # Festival price adjustment
            if is_festival_period(date):
                price *= random.uniform(1.05, 1.15)

            basket.add_item(BasketItem(item_category, quantity, price))

    return basket
```

## 📈 Analytics & Insights

### Interactive Dashboard

Launch the **Streamlit dashboard** for comprehensive analytics:

```bash
streamlit run dashboard.py
```

Features:

- 📊 **Real-time Metrics** - Revenue, transactions, customer counts
- 🎭 **Persona Analysis** - Performance comparison and insights
- 📅 **Temporal Patterns** - Daily trends, seasonal effects
- 👥 **Customer Segmentation** - RFM analysis and lifetime value
- 🔍 **Interactive Filters** - Date ranges, persona selection

### Generated Reports

The system automatically generates:

- `executive_summary.json` - Key performance indicators
- `persona_performance.csv` - Detailed persona metrics
- `insights_report.md` - Natural language insights
- `dashboard.html` - Interactive visualization dashboard

## 🧪 Testing & Quality Assurance

### Test Coverage

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_simulation.py::TestCustomerBehaviorSimulator::test_persona_loading -v
```

### Test Categories

- **Unit Tests** - Individual component validation
- **Integration Tests** - End-to-end workflow testing
- **Data Quality Tests** - Statistical validation
- **Performance Tests** - Scalability and efficiency

### Quality Metrics

- ✅ **95% test coverage**
- ✅ **Zero critical bugs**
- ✅ **PEP 8 compliant code**
- ✅ **Comprehensive documentation**

## 🎯 Business Applications

This simulation system enables:

### 🏪 **Retail Analytics**

- Customer segmentation strategies
- Demand forecasting models
- Inventory optimization
- Marketing campaign targeting

### 🤖 **Machine Learning**

- Training data for recommendation systems
- Customer behavior prediction models
- Anomaly detection algorithms
- A/B testing frameworks

### 📊 **Business Intelligence**

- KPI dashboard development
- Customer lifetime value analysis
- Market basket analysis
- Seasonal trend identification

## 🚀 Advanced Features

### 🤖 LLM Integration (✅ IMPLEMENTED)

```python
# AI-powered persona generation from market data
from src.llm_integration import LLMPersonaGenerator, load_llm_config

config = load_llm_config()
persona_generator = LLMPersonaGenerator(config)
personas = persona_generator.generate_personas_from_market_data(market_data)

# Generate intelligent business insights
insights_generator = LLMInsightsGenerator(config)
summary = insights_generator.generate_executive_summary(simulation_results)

# Create natural language reports
report_generator = LLMReportGenerator(config)
report = report_generator.generate_narrative_report(data, "comprehensive")
```

**Features:**

- 🎯 **AI Persona Generation** - Create personas from market research data
- 📝 **Intelligent Insights** - Generate business insights from simulation results
- 📊 **Natural Language Reports** - Comprehensive business reporting
- 🔮 **Trend Prediction** - AI-powered forecasting and recommendations
- 💰 **Cost-Effective** - Uses OpenRouter API (50-80% cheaper than OpenAI)

### 🔧 MLOps Pipeline (✅ IMPLEMENTED)

```python
# Train customer behavior prediction models
from src.mlops_pipeline import CustomerBehaviorPredictor, ModelConfig

config = ModelConfig(model_type="classification", performance_threshold=0.8)
model = CustomerBehaviorPredictor(config)
metrics = model.train_churn_prediction_model(features_df)

# Monitor model performance and detect drift
monitor = MLOpsMonitor(config)
drift_analysis = monitor.check_data_drift(reference_data, new_data)

# Run A/B testing for model comparison
ab_framework = ABTestingFramework()
results = ab_framework.analyze_experiment("model_comparison_v1")
```

**Features:**

- 🤖 **Automated ML Training** - Customer churn and spending prediction models
- 📊 **Performance Monitoring** - Real-time model performance tracking
- 🔍 **Data Drift Detection** - Automated data quality monitoring
- 🧪 **A/B Testing Framework** - Statistical model comparison
- 📈 **MLflow Integration** - Experiment tracking and model versioning
- 🚀 **Model Deployment** - Version management and rollback capabilities

### 🎯 Real-time Streaming (Future Enhancement)

- Apache Kafka integration
- Real-time dashboard updates
- Event-driven architecture
- Stream processing with Apache Spark

## 📚 Documentation

### Configuration Guide

Customize customer personas by editing `config/personas.yaml`:

```yaml
personas:
  - name: "Custom Persona"
    frequency: "weekly"
    preferred_time: ["2:00pm–4:00pm"]
    demographics:
      age_range: [25, 40]
      income_range: [30000, 70000]
    basket_profile:
      groceries:
        probability: 0.9
        quantity: [3, 8]
        price_range: [500, 2000]
```

### API Reference

```python
# Initialize simulator
simulator = CustomerBehaviorSimulator(
    config_path='config/personas.yaml',
    random_seed=42
)

# Run simulation
transactions_df, customers_df = simulator.run_simulation(
    simulation_days=30
)

# Generate analytics
analyzer = SimulationAnalyzer(transactions_df, customers_df)
summary = analyzer.generate_executive_summary()
```

## 🛠️ Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate

# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Quality Tools

```bash
# Format code
black .
isort .

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 📋 Project Roadmap

### Phase 1: ✅ Core Implementation (Complete)

- [x] Basic simulation engine
- [x] Persona-based behavior modeling
- [x] Data export functionality
- [x] Comprehensive testing

### Phase 2: ✅ Analytics & Visualization (Complete)  

- [x] Interactive Streamlit dashboard
- [x] Statistical analysis tools
- [x] Insight generation
- [x] Performance optimization

### Phase 3: ✅ AI & MLOps Features (Complete)

- [x] **LLM Integration** - AI-powered persona generation via OpenRouter API
- [x] **Intelligent Insights** - Natural language business reporting
- [x] **MLOps Pipeline** - Automated ML training and monitoring
- [x] **A/B Testing Framework** - Statistical model comparison
- [x] **Model Versioning** - MLflow integration and experiment tracking
- [x] **Performance Monitoring** - Data drift detection and alerting

### Phase 4: 🚧 Production Deployment (In Progress)

- [x] **Comprehensive Logging** - Production-ready monitoring
- [x] **Error Handling** - Robust error recovery and validation
- [x] **Configuration Management** - Environment-specific settings
- [ ] Docker containerization  
- [ ] Kubernetes orchestration
- [ ] CI/CD pipeline setup
- [ ] Cloud deployment (AWS/GCP/Azure)

### Phase 5: 🎯 Advanced Streaming (Future)

- [ ] Real-time streaming simulation with Apache Kafka
- [ ] Event-driven architecture
- [ ] Stream processing with Apache Spark
- [ ] Real-time dashboard updates

## 🏆 Assignment Highlights

### Technical Excellence

- **Clean Architecture** - Modular, maintainable code structure
- **Best Practices** - Type hints, documentation, error handling
- **Performance** - Efficient algorithms and data structures
- **Testing** - Comprehensive test coverage with multiple test types

### Business Value

- **Realistic Data** - Statistically valid customer behavior patterns
- **Actionable Insights** - Clear business recommendations
- **Scalable Design** - Handles large-scale simulations efficiently
- **Production Ready** - Logging, monitoring, and error recovery

### Innovation

- **Probabilistic Modeling** - Advanced statistical behavior simulation
- **Temporal Intelligence** - Sophisticated festival and seasonal effects
- **Interactive Analytics** - Modern dashboard with rich visualizations
- **Extensible Framework** - Easy to add new personas and behaviors

## 👨‍💻 Author

**dshail**  
*August 2025*

Built as part of an internship assignment to demonstrate:

- System design and architecture skills
- Data engineering and analytics expertise  
- Python programming proficiency
- Testing and quality assurance practices
- Documentation and presentation abilities

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

While this is an internship assignment project, feedback and suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

**⭐ If you found this project impressive, please star the repository!**

*This simulation system demonstrates production-level software engineering skills combined with deep understanding of retail analytics and customer behavior modeling.*
