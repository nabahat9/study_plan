"""
TRAIN SCRIPT - Run this ONCE to train and save ECLAT models
python train.py
"""
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os
from study_planner import StudyPlanner

def train_and_save():
    """Train ECLAT model and save it"""
    print("🚀 Starting model training...")
    
    # Initialize planner
    planner = StudyPlanner("data/study_plans.csv")
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df = planner.load_and_preprocess()
    
    # Run ECLAT algorithm
    print("🔍 Training ECLAT model...")
    rules = planner.run_eclat(min_support=0.15, min_confidence=0.6)
    
    # Create metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(df),
        'features_count': len(df.columns),
        'rules_count': len(rules),
        'min_support': 0.15,
        'min_confidence': 0.6,
        'algorithm': 'ECLAT'
    }
    
    # Save trained model
    os.makedirs('trained_models', exist_ok=True)
    
    # Save rules
    with open('trained_models/eclat_rules.pkl', 'wb') as f:
        pickle.dump(rules, f)
    
    # Save metadata
    with open('trained_models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save processed data (optional, for debugging)
    with open('trained_models/processed_data.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    print(f"✅ Training complete!")
    print(f"   - Saved {len(rules)} association rules")
    print(f"   - Metadata saved to trained_models/metadata.json")
    print(f"   - Model ready for FastAPI server")

if __name__ == "__main__":
    train_and_save()