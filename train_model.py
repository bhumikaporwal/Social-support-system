#!/usr/bin/env python3
"""
Simple script to train and test the ML eligibility model
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from data.synthetic.generate_test_data import SyntheticDataGenerator
from src.ml.model_trainer import EligibilityModelTrainer

def main():
    print("Starting ML model training...")

    # Step 1: Generate fresh training data
    print("\nStep 1: Generating training data...")
    generator = SyntheticDataGenerator()
    generator.save_test_data(num_applications=800)
    print("Training data generated successfully")

    # Step 2: Train ML models
    print("\nStep 2: Training ML models...")
    trainer = EligibilityModelTrainer()

    # Load and prepare data
    df = trainer.load_training_data()
    print(f"Loaded {len(df)} training samples")

    # Train models
    results = trainer.train_models(df)

    # Save trained models
    trainer.save_models()

    # Evaluate best model
    evaluation = trainer.evaluate_model()

    print(f"\nBest Model: {evaluation['model_name']}")
    print(f"Accuracy: {evaluation['accuracy']:.3f}")
    print(f"Features used: {evaluation['feature_count']}")

    print(f"\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(evaluation['top_features'][:10], 1):
        print(f"   {i:2d}. {feature}: {importance:.3f}")

    print("\nTraining completed successfully!")
    print("Models saved to 'models/' directory")

if __name__ == "__main__":
    main()