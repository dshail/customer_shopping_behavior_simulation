# 🔑 OpenRouter API Setup Guide

## ✅ **All API Code Updated to OpenRouter**

The system has been **completely migrated** from OpenAI to OpenRouter API. Here's what was changed:

### 🔄 **Code Changes Made:**

1. **LLM Integration Module** (`src/llm_integration.py`):
   - ✅ Replaced OpenAI client with direct HTTP requests to OpenRouter
   - ✅ Updated all API calls to use OpenRouter endpoints
   - ✅ Added proper OpenRouter headers (Authorization, HTTP-Referer, X-Title)
   - ✅ Changed default model to `anthropic/claude-3-haiku` (cost-effective)

2. **Configuration Files**:
   - ✅ Updated `config/llm_config.yaml` with OpenRouter settings
   - ✅ Added multiple model options (Anthropic, OpenAI, Google, etc.)
   - ✅ Included cost management and routing preferences

3. **Dependencies** (`requirements.txt`):
   - ✅ Removed `openai>=1.0.0` and `tiktoken>=0.5.0`
   - ✅ Added `requests>=2.31.0` for HTTP API calls

4. **Demo and Main Scripts**:
   - ✅ Updated all references from OpenAI to OpenRouter
   - ✅ Changed API key validation checks
   - ✅ Updated installation instructions

## 🔑 **Where to Update Your API Key**

### **Method 1: Configuration File (Recommended)**

Edit `config/llm_config.yaml`:

```yaml
# Replace this line:
api_key: "your-openrouter-api-key-here"

# With your actual API key:
api_key: "sk-or-v1-your-actual-api-key-here"
```

### **Method 2: Environment Variable**

Set the environment variable:

```bash
# Linux/Mac
export OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"

# Windows Command Prompt
set OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here

# Windows PowerShell
$env:OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"
```

## 🚀 **How to Get OpenRouter API Key**

### **Step 1: Sign Up**
1. Go to [https://openrouter.ai/](https://openrouter.ai/)
2. Click "Sign Up" or "Login"
3. Create account with email/Google/GitHub

### **Step 2: Get API Key**
1. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Click "Create Key"
3. Give it a name (e.g., "Customer Behavior Simulation")
4. Copy the key (starts with `sk-or-v1-`)

### **Step 3: Add Credits (Optional)**
1. Go to [https://openrouter.ai/credits](https://openrouter.ai/credits)
2. Add $5-10 to start (very cost-effective)
3. OpenRouter is typically 50-80% cheaper than direct OpenAI

## 💰 **Cost Comparison & Model Selection**

### **Recommended Models by Use Case:**

#### **🏆 Best Value (Recommended)**
```yaml
model: "anthropic/claude-3-haiku"
# Cost: ~$0.25/1M input tokens, $1.25/1M output tokens
# Speed: Very fast
# Quality: Excellent for most tasks
```

#### **⚖️ Balanced Performance**
```yaml
model: "openai/gpt-3.5-turbo"
# Cost: ~$0.50/1M input tokens, $1.50/1M output tokens
# Speed: Fast
# Quality: Good for complex tasks
```

#### **🎯 Premium Quality**
```yaml
model: "anthropic/claude-3-sonnet"
# Cost: ~$3/1M input tokens, $15/1M output tokens
# Speed: Medium
# Quality: Excellent for complex analysis
```

#### **🚀 Maximum Performance**
```yaml
model: "openai/gpt-4"
# Cost: ~$30/1M input tokens, $60/1M output tokens
# Speed: Slower
# Quality: Best for complex reasoning
```

## ⚙️ **Configuration Examples**

### **Basic Configuration**
```yaml
# config/llm_config.yaml
api_key: "sk-or-v1-your-key-here"
base_url: "https://openrouter.ai/api/v1"
model: "anthropic/claude-3-haiku"
temperature: 0.7
max_tokens: 2000
```

### **Advanced Configuration**
```yaml
# config/llm_config.yaml
api_key: "sk-or-v1-your-key-here"
base_url: "https://openrouter.ai/api/v1"
model: "anthropic/claude-3-haiku"
temperature: 0.7
max_tokens: 2000
site_url: "https://github.com/your-username/customer-behavior-sim"
site_name: "Customer Behavior Simulation"

# Task-specific models
task_models:
  persona_generation:
    model: "anthropic/claude-3-sonnet"  # Better creativity
    temperature: 0.8
    max_tokens: 3000
  
  insights_generation:
    model: "anthropic/claude-3-haiku"   # Cost-effective analysis
    temperature: 0.3
    max_tokens: 1500

# Cost management
cost_management:
  max_monthly_spend: 25.0  # USD
  alert_threshold: 20.0    # USD
  track_usage: true
```

## 🧪 **Testing Your Setup**

### **Quick Test**
```bash
# Run the demo to test LLM features
python demo_enhanced_features.py
```

### **Manual Test**
```python
from src.llm_integration import load_llm_config, LLMPersonaGenerator

# Load config
config = load_llm_config()
print(f"API Key configured: {config.api_key != 'your-openrouter-api-key-here'}")
print(f"Model: {config.model}")

# Test API call
generator = LLMPersonaGenerator(config)
# This will make a real API call if key is configured
```

## 🔍 **Troubleshooting**

### **Common Issues & Solutions**

#### **❌ "API key not configured"**
```bash
# Check your config file
cat config/llm_config.yaml | grep api_key

# Or check environment variable
echo $OPENROUTER_API_KEY
```

#### **❌ "HTTP 401 Unauthorized"**
- Verify your API key is correct
- Check if you have credits in your OpenRouter account
- Ensure key starts with `sk-or-v1-`

#### **❌ "HTTP 429 Rate Limited"**
- Reduce `requests_per_minute` in config
- Add delays between requests
- Upgrade your OpenRouter plan

#### **❌ "Model not found"**
- Check available models at [https://openrouter.ai/models](https://openrouter.ai/models)
- Verify model name spelling
- Some models require special access

#### **❌ "Import Error: requests"**
```bash
# Install required dependencies
pip install requests>=2.31.0
```

## 📊 **Monitoring Usage & Costs**

### **OpenRouter Dashboard**
1. Go to [https://openrouter.ai/activity](https://openrouter.ai/activity)
2. Monitor your usage and costs
3. Set up alerts for spending limits

### **In-App Monitoring**
```python
# The system logs all API calls
# Check logs for usage patterns
tail -f logs/enhanced_simulation_*.log | grep "API request"
```

## 🎯 **Best Practices**

### **Cost Optimization**
1. **Start with cheaper models** (`claude-3-haiku`)
2. **Use task-specific models** (different models for different tasks)
3. **Set spending limits** in configuration
4. **Cache responses** when possible
5. **Use lower temperatures** for factual tasks (0.1-0.3)

### **Performance Optimization**
1. **Batch requests** when possible
2. **Use appropriate max_tokens** (don't over-allocate)
3. **Implement retry logic** with exponential backoff
4. **Monitor response times** and switch models if needed

### **Security**
1. **Never commit API keys** to version control
2. **Use environment variables** in production
3. **Rotate keys regularly**
4. **Monitor for unusual usage**

## 🚀 **Ready to Use!**

Your system is now fully configured for OpenRouter! Here's what you can do:

### **1. Update Your API Key**
```bash
# Edit the config file
nano config/llm_config.yaml
# Replace: api_key: "your-openrouter-api-key-here"
# With: api_key: "sk-or-v1-your-actual-key"
```

### **2. Test the Integration**
```bash
# Run the demo
python demo_enhanced_features.py

# Or run the full enhanced simulation
python main_enhanced.py --enable-ai-insights
```

### **3. Explore Features**
- 🤖 **AI Persona Generation**: Create personas from market data
- 🧠 **Intelligent Insights**: Generate business insights from simulation results
- 📝 **Natural Language Reports**: Create comprehensive business reports
- 🔮 **Trend Prediction**: Predict future shopping behaviors

## 💡 **Why OpenRouter?**

### **Advantages over Direct OpenAI:**
1. **💰 Cost Savings**: 50-80% cheaper than direct OpenAI
2. **🔄 Model Variety**: Access to Anthropic, Google, Meta, and more
3. **🛡️ Reliability**: Automatic failover between providers
4. **📊 Unified API**: One API for multiple LLM providers
5. **🎯 Model Routing**: Automatic selection of best model for task

### **Perfect for This Project:**
- **Persona Generation**: Claude-3 excels at creative tasks
- **Data Analysis**: Multiple models for different analytical needs
- **Cost Control**: Essential for simulation workloads
- **Scalability**: Handle varying loads efficiently

---

## 🎉 **You're All Set!**

The Customer Behavior Simulation System now uses OpenRouter for:
- ✅ **AI-powered persona generation**
- ✅ **Intelligent business insights**
- ✅ **Natural language reporting**
- ✅ **Predictive analytics**

**Next Steps:**
1. Get your OpenRouter API key
2. Update `config/llm_config.yaml`
3. Run `python demo_enhanced_features.py`
4. Explore the enhanced simulation features!

**Need Help?** Check the logs in `logs/` directory or run the demo script for detailed error messages.