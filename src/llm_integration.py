"""
LLM Integration Module for Customer Behavior Simulation

This module provides AI-powered features including:
- Persona generation from market data
- Intelligent insights generation
- Natural language reporting
- Behavior pattern analysis

Uses OpenRouter API for access to multiple LLM providers
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
from datetime import datetime
import pandas as pd
import yaml

from .models import PersonaConfig, Demographics


@dataclass
class LLMConfig:
    """Configuration for LLM integration via OpenRouter"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3-haiku"  # Default to cost-effective model
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    site_url: str = "https://github.com/your-repo"  # For OpenRouter attribution
    site_name: str = "Customer Behavior Simulation"


class LLMPersonaGenerator:
    """AI-powered persona generation from market data"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.site_url,
            "X-Title": config.site_name
        })
        self.logger = logging.getLogger(__name__)
    
    def _make_api_request(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Make API request to OpenRouter"""
        try:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def generate_personas_from_market_data(
        self, 
        market_data: Dict[str, Any],
        num_personas: int = 5
    ) -> List[PersonaConfig]:
        """Generate customer personas based on market research data"""
        
        prompt = self._build_persona_generation_prompt(market_data, num_personas)
        
        try:
            response = self._make_api_request([
                {"role": "system", "content": "You are an expert retail analyst and customer behavior specialist."},
                {"role": "user", "content": prompt}
            ])
            
            if response and 'choices' in response:
                personas_data = json.loads(response['choices'][0]['message']['content'])
                personas = []
                
                for persona_data in personas_data.get('personas', []):
                    persona = self._convert_to_persona_config(persona_data)
                    personas.append(persona)
                
                self.logger.info(f"Generated {len(personas)} personas using LLM")
                return personas
            
        except Exception as e:
            self.logger.error(f"Failed to generate personas: {e}")
            return []
        
        return []
    
    def enhance_existing_persona(
        self, 
        persona: PersonaConfig, 
        market_trends: Dict[str, Any]
    ) -> PersonaConfig:
        """Enhance existing persona with market trend insights"""
        
        prompt = f"""
        Analyze and enhance this customer persona based on current market trends:
        
        Current Persona: {asdict(persona)}
        Market Trends: {market_trends}
        
        Provide enhanced persona configuration with:
        1. Updated shopping behaviors
        2. New product preferences
        3. Adjusted spending patterns
        4. Seasonal variations
        
        Return as valid YAML configuration.
        """
        
        try:
            response = self._make_api_request([
                {"role": "system", "content": "You are a customer behavior expert."},
                {"role": "user", "content": prompt}
            ])
            
            if response and 'choices' in response:
                enhanced_data = yaml.safe_load(response['choices'][0]['message']['content'])
                enhanced_persona = self._convert_to_persona_config(enhanced_data)
                
                self.logger.info(f"Enhanced persona: {persona.name}")
                return enhanced_persona
            
        except Exception as e:
            self.logger.error(f"Failed to enhance persona {persona.name}: {e}")
        
        return persona
    
    def _build_persona_generation_prompt(
        self, 
        market_data: Dict[str, Any], 
        num_personas: int
    ) -> str:
        """Build prompt for persona generation"""
        
        return f"""
        Generate {num_personas} distinct customer personas for a grocery retail simulation based on this market data:
        
        Market Data: {json.dumps(market_data, indent=2)}
        
        For each persona, provide:
        1. Name (descriptive, e.g., "Tech-Savvy Millennial")
        2. Demographics (age_range, income_range, location_type)
        3. Shopping frequency (daily, weekly, monthly, etc.)
        4. Preferred shopping times
        5. Basket profile with product categories and probabilities
        6. Price sensitivity and spending patterns
        7. Seasonal behavior variations
        
        Return as JSON with this structure:
        {{
            "personas": [
                {{
                    "name": "Persona Name",
                    "frequency": "weekly",
                    "preferred_time": ["10:00am-12:00pm"],
                    "demographics": {{
                        "age_range": [25, 40],
                        "income_range": [40000, 80000]
                    }},
                    "basket_profile": {{
                        "groceries": {{
                            "probability": 0.9,
                            "quantity": [3, 8],
                            "price_range": [500, 2000]
                        }}
                    }}
                }}
            ]
        }}
        """
    
    def _convert_to_persona_config(self, persona_data: Dict[str, Any]) -> PersonaConfig:
        """Convert LLM-generated data to PersonaConfig object"""
        
        demographics = Demographics(
            age_range=tuple(persona_data['demographics']['age_range']),
            income_range=tuple(persona_data['demographics']['income_range'])
        )
        
        return PersonaConfig(
            name=persona_data['name'],
            frequency=persona_data['frequency'],
            preferred_time=persona_data['preferred_time'],
            demographics=demographics,
            basket_profile=persona_data['basket_profile']
        )


class LLMInsightsGenerator:
    """AI-powered insights and reporting"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.site_url,
            "X-Title": config.site_name
        })
        self.logger = logging.getLogger(__name__)
    
    def _make_api_request(self, messages: List[Dict[str, str]], temperature: float = None) -> Optional[Dict[str, Any]]:
        """Make API request to OpenRouter"""
        try:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def generate_executive_summary(
        self, 
        simulation_results: Dict[str, Any]
    ) -> str:
        """Generate executive summary with AI insights"""
        
        prompt = f"""
        Analyze these customer behavior simulation results and generate an executive summary:
        
        Simulation Results: {json.dumps(simulation_results, indent=2)}
        
        Provide:
        1. Key performance highlights
        2. Customer behavior insights
        3. Revenue optimization opportunities
        4. Market trends identified
        5. Strategic recommendations
        6. Risk factors and considerations
        
        Write in professional business language suitable for C-level executives.
        """
        
        try:
            response = self._make_api_request([
                {"role": "system", "content": "You are a senior retail analytics consultant."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)  # Lower temperature for more factual content
            
            if response and 'choices' in response:
                summary = response['choices'][0]['message']['content']
                self.logger.info("Generated executive summary using LLM")
                return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
        
        return "Executive summary generation failed."
    
    def analyze_customer_segments(
        self, 
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """AI-powered customer segmentation analysis"""
        
        # Prepare data summary for LLM
        segment_data = {
            "total_customers": len(customers_df),
            "total_transactions": len(transactions_df),
            "revenue_by_persona": transactions_df.groupby('persona_type')['total_amount'].sum().to_dict(),
            "avg_transaction_by_persona": transactions_df.groupby('persona_type')['total_amount'].mean().to_dict(),
            "transaction_frequency": transactions_df.groupby('customer_id').size().describe().to_dict()
        }
        
        prompt = f"""
        Analyze this customer segmentation data and provide insights:
        
        Segment Data: {json.dumps(segment_data, indent=2)}
        
        Provide analysis on:
        1. High-value customer characteristics
        2. Customer lifetime value patterns
        3. Churn risk indicators
        4. Cross-selling opportunities
        5. Personalization strategies
        6. Retention recommendations
        
        Return as structured JSON with actionable insights.
        """
        
        try:
            response = self._make_api_request([
                {"role": "system", "content": "You are a customer analytics expert."},
                {"role": "user", "content": prompt}
            ], temperature=0.4)
            
            if response and 'choices' in response:
                insights = json.loads(response['choices'][0]['message']['content'])
                self.logger.info("Generated customer segmentation insights")
                return insights
            
        except Exception as e:
            self.logger.error(f"Failed to analyze customer segments: {e}")
        
        return {}
    
    def predict_future_trends(
        self, 
        historical_data: Dict[str, Any],
        external_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict future shopping trends using AI"""
        
        prompt = f"""
        Based on historical shopping data and external factors, predict future trends:
        
        Historical Data: {json.dumps(historical_data, indent=2)}
        External Factors: {json.dumps(external_factors, indent=2)}
        
        Predict:
        1. Emerging customer behaviors
        2. Product category growth/decline
        3. Seasonal pattern changes
        4. Technology adoption impacts
        5. Economic factor influences
        6. Competitive landscape effects
        
        Provide confidence levels and timeframes for predictions.
        """
        
        try:
            response = self._make_api_request([
                {"role": "system", "content": "You are a retail futurist and trend analyst."},
                {"role": "user", "content": prompt}
            ], temperature=0.6)
            
            if response and 'choices' in response:
                predictions = json.loads(response['choices'][0]['message']['content'])
                self.logger.info("Generated future trend predictions")
                return predictions
            
        except Exception as e:
            self.logger.error(f"Failed to predict future trends: {e}")
        
        return {}


class LLMReportGenerator:
    """Generate natural language reports from simulation data"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.site_url,
            "X-Title": config.site_name
        })
        self.logger = logging.getLogger(__name__)
    
    def _make_api_request(self, messages: List[Dict[str, str]], temperature: float = None, max_tokens: int = None) -> Optional[Dict[str, Any]]:
        """Make API request to OpenRouter"""
        try:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens
            }
            
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def generate_narrative_report(
        self, 
        simulation_data: Dict[str, Any],
        report_type: str = "comprehensive"
    ) -> str:
        """Generate narrative report from simulation data"""
        
        report_prompts = {
            "comprehensive": "Generate a comprehensive business report",
            "executive": "Generate an executive briefing",
            "technical": "Generate a technical analysis report",
            "marketing": "Generate a marketing insights report"
        }
        
        base_prompt = report_prompts.get(report_type, report_prompts["comprehensive"])
        
        prompt = f"""
        {base_prompt} based on this customer behavior simulation data:
        
        Simulation Data: {json.dumps(simulation_data, indent=2)}
        
        Structure the report with:
        1. Executive Summary
        2. Key Findings
        3. Customer Behavior Analysis
        4. Revenue Performance
        5. Market Opportunities
        6. Recommendations
        7. Next Steps
        
        Use professional business language with data-driven insights.
        """
        
        try:
            response = self._make_api_request([
                {"role": "system", "content": f"You are a senior business analyst creating a {report_type} report."},
                {"role": "user", "content": prompt}
            ], temperature=0.3, max_tokens=2000)
            
            if response and 'choices' in response:
                report = response['choices'][0]['message']['content']
                self.logger.info(f"Generated {report_type} narrative report")
                return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate {report_type} report: {e}")
        
        return f"Report generation failed for {report_type} report."
    
    def generate_persona_insights(
        self, 
        persona_performance: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate insights for each persona"""
        
        insights = {}
        
        for persona_name, performance_data in persona_performance.items():
            prompt = f"""
            Analyze this customer persona performance and provide insights:
            
            Persona: {persona_name}
            Performance Data: {json.dumps(performance_data, indent=2)}
            
            Provide:
            1. Behavioral characteristics
            2. Revenue contribution analysis
            3. Optimization opportunities
            4. Marketing recommendations
            5. Retention strategies
            
            Keep response concise and actionable.
            """
            
            try:
                response = self._make_api_request([
                    {"role": "system", "content": "You are a customer behavior specialist."},
                    {"role": "user", "content": prompt}
                ], temperature=0.4, max_tokens=800)
                
                if response and 'choices' in response:
                    insights[persona_name] = response['choices'][0]['message']['content']
                else:
                    insights[persona_name] = f"Insight generation failed for {persona_name}"
                
            except Exception as e:
                self.logger.error(f"Failed to generate insights for {persona_name}: {e}")
                insights[persona_name] = f"Insight generation failed for {persona_name}"
        
        return insights


# Utility functions for LLM integration
def load_llm_config(config_path: str = "config/llm_config.yaml") -> Optional[LLMConfig]:
    """Load LLM configuration from file"""
    try:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        return LLMConfig(
            api_key=config_data.get('api_key', os.getenv('OPENROUTER_API_KEY')),
            base_url=config_data.get('base_url', 'https://openrouter.ai/api/v1'),
            model=config_data.get('model', 'anthropic/claude-3-haiku'),
            temperature=config_data.get('temperature', 0.7),
            max_tokens=config_data.get('max_tokens', 2000),
            timeout=config_data.get('timeout', 30),
            site_url=config_data.get('site_url', 'https://github.com/your-repo'),
            site_name=config_data.get('site_name', 'Customer Behavior Simulation')
        )
    except Exception as e:
        logging.error(f"Failed to load LLM config: {e}")
        return None


def create_sample_llm_config():
    """Create sample LLM configuration file for OpenRouter"""
    sample_config = {
        'api_key': 'your-openrouter-api-key-here',
        'base_url': 'https://openrouter.ai/api/v1',
        'model': 'anthropic/claude-3-haiku',  # Cost-effective default
        'temperature': 0.7,
        'max_tokens': 2000,
        'timeout': 30,
        'site_url': 'https://github.com/your-repo',
        'site_name': 'Customer Behavior Simulation',
        'features': {
            'persona_generation': True,
            'insights_generation': True,
            'report_generation': True,
            'trend_prediction': True
        },
        'model_options': {
            'cost_effective': 'anthropic/claude-3-haiku',
            'balanced': 'openai/gpt-3.5-turbo',
            'premium': 'openai/gpt-4',
            'advanced': 'anthropic/claude-3-opus'
        }
    }
    
    os.makedirs('config', exist_ok=True)
    with open('config/llm_config.yaml', 'w') as file:
        yaml.dump(sample_config, file, default_flow_style=False)
    
    print("Sample LLM configuration created at config/llm_config.yaml")
    print("Please update with your OpenRouter API key from https://openrouter.ai/")