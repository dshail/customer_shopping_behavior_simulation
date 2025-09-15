#!/usr/bin/env python3
"""
Debug script for ML model training issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.simulator import CustomerBehaviorSimulator
from src.mlops_pipeline import CustomerBehaviorPredictor, ModelConfig

def debug_column_structure():
    """Debug the column structure issue in ML training"""
    
    print("🔍 Debugging ML Model Training Issues...")
    
    # Initialize simulator
    simulator = CustomerBehaviorSimulator()
    
    # Run a small simulation
    print("Running small simulation...")
    transactions_df, customers_df = simulator.run_simulation(simulation_days=5)
    
    print(f"Generated {len(transactions_df)} transactions, {len(customers_df)} customers")
    
    # Debug column structure
    print("\n📊 Debugging Column Structure:")
    print("Transactions columns:", transactions_df.columns.tolist())
    print("Customers columns:", customers_df.columns.tolist())
    
    # Test feature preparation
    print("\n🔧 Testing Feature Preparation:")
    
    try:
        # Create model config
        model_config = ModelConfig(
            model_type="classification",
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
        
        # Create model
        model = CustomerBehaviorPredictor(model_config)
        
        # Test prepare_features
        print("Calling prepare_features...")
        features_df = model.prepare_features(transactions_df, customers_df)
        
        print("✅ Feature preparation successful!")
        print(f"Features shape: {features_df.shape}")
        print("Feature columns:", features_df.columns.tolist())
        
        # Test groupby aggregation step by step
        print("\n🔍 Step-by-step debugging:")
        
        # Step 1: Groupby aggregation
        customer_features = transactions_df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'std', 'count'],
            'num_items': ['sum', 'mean'],
            'date': ['min', 'max']
        }).reset_index()
        
        print("After groupby - shape:", customer_features.shape)
        print("After groupby - columns:", customer_features.columns.tolist())
        print("Column structure:")
        for i, col in enumerate(customer_features.columns):
            print(f"  {i}: {col} (type: {type(col)})")
        
        # Step 2: Column flattening
        print("\n🔧 Testing column flattening:")
        try:
            new_columns = ['_'.join(col).strip() if col[1] else col[0] for col in customer_features.columns.values]
            print("New columns:", new_columns)
            customer_features.columns = new_columns
            print("✅ Column flattening successful!")
        except Exception as e:
            print(f"❌ Column flattening failed: {e}")
            print("Trying alternative approach...")
            
            # Alternative approach
            flat_columns = []
            for col in customer_features.columns:
                if isinstance(col, tuple):
                    if col[1]:  # If second element exists and is not empty
                        flat_columns.append(f"{col[0]}_{col[1]}")
                    else:
                        flat_columns.append(col[0])
                else:
                    flat_columns.append(str(col))
            
            print("Alternative columns:", flat_columns)
            customer_features.columns = flat_columns
            print("✅ Alternative column flattening successful!")
        
        print("Final columns:", customer_features.columns.tolist())
        
    except Exception as e:
        print(f"❌ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_column_structure()