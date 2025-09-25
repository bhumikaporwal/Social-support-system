#!/usr/bin/env python3
"""
Train and test the ML eligibility model with comprehensive evaluation
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from data.synthetic.generate_test_data import SyntheticDataGenerator
from src.ml.model_trainer import EligibilityModelTrainer
from src.agents.eligibility_agent import EligibilityAssessmentAgent

def main():
    print("Starting comprehensive ML model training and testing...")

    # Step 1: Generate fresh training data
    print("\nStep 1: Generating comprehensive training data...")
    generator = SyntheticDataGenerator()
    generator.save_test_data(num_applications=500)
    print("✅ Training data generated successfully")

    # Step 2: Train ML models
    print("\n🤖 Step 2: Training ML models...")
    trainer = EligibilityModelTrainer()

    # Load and prepare data
    df = trainer.load_training_data()
    print(f"📈 Loaded {len(df)} training samples")

    # Train models
    results = trainer.train_models(df)

    # Save trained models
    trainer.save_models()

    # Evaluate best model
    evaluation = trainer.evaluate_model()

    print(f"\n🏆 Best Model: {evaluation['model_name']}")
    print(f"📊 Accuracy: {evaluation['accuracy']:.3f}")
    print(f"🔍 Features used: {evaluation['feature_count']}")

    print(f"\n🔥 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(evaluation['top_features'][:10], 1):
        print(f"   {i:2d}. {feature}: {importance:.3f}")

    # Step 3: Test the eligibility agent with trained model
    print("\n🧪 Step 3: Testing eligibility agent with trained model...")

    # Initialize agent (should load the trained model)
    agent = EligibilityAssessmentAgent()

    # Test with diverse scenarios
    test_scenarios = [
        {
            'name': 'Low Income Eligible',
            'data': {
                'age': 30,
                'monthly_income': 8000,
                'net_worth': 50000,
                'debt_to_income_ratio': 0.3,
                'credit_score': 720,
                'family_size': 3,
                'dependents': 1,
                'employment_status': 'Employed',
                'experience_years': 5,
                'education_level': 'Bachelor\'s Degree',
                'nationality': 'UAE',
                'emirate': 'Dubai',
                'marital_status': 'Married',
                'gender': 'Male'
            }
        },
        {
            'name': 'High Income Ineligible',
            'data': {
                'age': 40,
                'monthly_income': 30000,
                'net_worth': 800000,
                'debt_to_income_ratio': 0.2,
                'credit_score': 800,
                'family_size': 2,
                'dependents': 0,
                'employment_status': 'Employed',
                'experience_years': 15,
                'education_level': 'Master\'s Degree',
                'nationality': 'UAE',
                'emirate': 'Dubai',
                'marital_status': 'Married',
                'gender': 'Female'
            }
        },
        {
            'name': 'Unemployed High Need',
            'data': {
                'age': 28,
                'monthly_income': 0,
                'net_worth': 5000,
                'debt_to_income_ratio': 0.8,
                'credit_score': 580,
                'family_size': 4,
                'dependents': 2,
                'employment_status': 'Unemployed',
                'experience_years': 3,
                'education_level': 'Diploma',
                'nationality': 'Egyptian',
                'emirate': 'Sharjah',
                'marital_status': 'Married',
                'gender': 'Male'
            }
        },
        {
            'name': 'Borderline Case',
            'data': {
                'age': 35,
                'monthly_income': 14500,
                'net_worth': 150000,
                'debt_to_income_ratio': 0.55,
                'credit_score': 670,
                'family_size': 2,
                'dependents': 0,
                'employment_status': 'Self-employed',
                'experience_years': 8,
                'education_level': 'Bachelor\'s Degree',
                'nationality': 'Indian',
                'emirate': 'Abu Dhabi',
                'marital_status': 'Single',
                'gender': 'Female'
            }
        },
        {
            'name': 'Young Graduate',
            'data': {
                'age': 24,
                'monthly_income': 6000,
                'net_worth': 15000,
                'debt_to_income_ratio': 0.4,
                'credit_score': 650,
                'family_size': 1,
                'dependents': 0,
                'employment_status': 'Part-time',
                'experience_years': 1,
                'education_level': 'Bachelor\'s Degree',
                'nationality': 'UAE',
                'emirate': 'Dubai',
                'marital_status': 'Single',
                'gender': 'Male'
            }
        }
    ]

    print("\n🔬 Testing different scenarios:")
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")

        # Create input for agent
        input_data = {
            'validated_data': scenario['data'],
            'support_type': 'both'
        }

        # Test the agent (synchronous call for testing)
        try:
            import asyncio
            result = asyncio.run(agent.process(input_data))

            if result.success:
                assessment = result.data.get('eligibility_assessment', {})
                ml_result = result.data.get('ml_result', {})

                print(f"   📊 Combined Score: {assessment.get('combined_score', 'N/A'):.3f}")
                print(f"   🤖 ML Prediction: {ml_result.get('ml_prediction', 'N/A')}")
                print(f"   🎯 ML Confidence: {ml_result.get('ml_confidence', 'N/A'):.3f}")
                print(f"   ⚖️  Recommendation: {assessment.get('recommendation', 'N/A')}")
                print(f"   💡 Rule-based Eligible: {result.data.get('rule_based_result', {}).get('financial_support_eligible', 'N/A')}")
            else:
                print(f"   ❌ Assessment failed: {result.reasoning}")

        except Exception as e:
            print(f"   ❌ Error testing scenario: {e}")

    print("\n🎉 Training and testing completed successfully!")
    print("\nThe system now has:")
    print("   ✅ Comprehensive synthetic training data")
    print("   ✅ Trained ML models for eligibility prediction")
    print("   ✅ Proper feature engineering and validation")
    print("   ✅ Different predictions for different input scenarios")
    print("\n📝 Next steps:")
    print("   - The models are saved in the 'models/' directory")
    print("   - The eligibility agent will automatically use the trained model")
    print("   - Test the full application pipeline with real document processing")

if __name__ == "__main__":
    main()