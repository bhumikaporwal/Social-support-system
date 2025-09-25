import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EligibilityModelTrainer:
    """Train ML model for eligibility assessment with proper feature engineering"""

    def __init__(self, data_path: str = "data/synthetic"):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}

    def load_training_data(self) -> pd.DataFrame:
        """Load and prepare training data from synthetic dataset"""
        logger.info("Loading training data...")

        # Load complete applications data
        with open(self.data_path / 'complete_applications.json', 'r') as f:
            applications = json.load(f)

        # Extract features for ML training
        training_data = []

        for app in applications:
            try:
                features = self._extract_features(app)
                if features:  # Only add if feature extraction succeeded
                    training_data.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for {app.get('application_id', 'unknown')}: {e}")
                continue

        df = pd.DataFrame(training_data)
        logger.info(f"Loaded {len(df)} training samples with {len(df.columns)} features")

        return df

    def _extract_features(self, app: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from application data"""
        applicant = app['applicant']
        bank_data = app['documents']['bank_statement']
        resume_data = app['documents']['resume']
        assets_data = app['documents']['assets_liabilities']
        credit_data = app['documents']['credit_report']
        expected = app['expected_outcome']

        # Calculate additional derived features
        monthly_income = bank_data['monthly_income']
        monthly_expenses = bank_data['monthly_expenses']
        net_worth = assets_data['net_worth']
        family_size = applicant['family_size']

        # Per-capita income (important for family need assessment)
        per_capita_income = monthly_income / family_size if family_size > 0 else monthly_income

        # Savings rate
        savings_rate = (monthly_income - monthly_expenses) / monthly_income if monthly_income > 0 else 0

        # Debt service ratio
        debt_payments = credit_data.get('monthly_obligations', 0)
        debt_service_ratio = debt_payments / monthly_income if monthly_income > 0 else 0

        # Asset liquidity ratio
        liquid_assets = assets_data.get('liquid_assets', 0)
        asset_liquidity_ratio = liquid_assets / assets_data['total_assets'] if assets_data['total_assets'] > 0 else 0

        # Experience to age ratio
        experience_years = resume_data.get('total_experience_years', 0)
        age = applicant['age']
        experience_ratio = experience_years / (age - 18) if age > 18 else 0

        features = {
            # Target variable
            'eligibility_score': expected['eligibility_score'],
            'expected_decision': expected['expected_decision'],

            # Basic demographics
            'age': age,
            'gender_male': 1 if applicant['gender'] == 'Male' else 0,
            'family_size': family_size,
            'dependents': applicant['dependents'],

            # Location factors
            'emirate_dubai': 1 if applicant['emirate'] == 'Dubai' else 0,
            'emirate_abu_dhabi': 1 if applicant['emirate'] == 'Abu Dhabi' else 0,
            'high_cost_emirate': 1 if applicant['emirate'] in ['Dubai', 'Abu Dhabi'] else 0,

            # Nationality (UAE nationals may have different criteria)
            'uae_national': 1 if applicant['nationality'] == 'UAE' else 0,

            # Education features
            'education_level': self._encode_education(applicant['education_level']),
            'has_degree': 1 if 'Degree' in applicant['education_level'] else 0,
            'has_advanced_degree': 1 if applicant['education_level'] in ['Master\'s Degree', 'PhD'] else 0,

            # Financial features
            'monthly_income': monthly_income,
            'monthly_expenses': monthly_expenses,
            'monthly_net': monthly_income - monthly_expenses,
            'per_capita_income': per_capita_income,
            'savings_rate': savings_rate,

            # Income categories for eligibility thresholds
            'income_eligible_financial': 1 if monthly_income <= 15000 else 0,
            'income_eligible_economic': 1 if monthly_income <= 25000 else 0,
            'very_low_income': 1 if monthly_income <= 5000 else 0,
            'low_income': 1 if 5000 < monthly_income <= 10000 else 0,
            'medium_income': 1 if 10000 < monthly_income <= 20000 else 0,

            # Wealth and assets
            'net_worth': net_worth,
            'total_assets': assets_data['total_assets'],
            'total_liabilities': assets_data['total_liabilities'],
            'liquid_assets': liquid_assets,
            'asset_liquidity_ratio': asset_liquidity_ratio,
            'net_worth_eligible': 1 if net_worth <= 500000 else 0,

            # Credit and debt
            'credit_score': credit_data['credit_score'],
            'debt_to_income_ratio': bank_data.get('debt_to_income_ratio', debt_service_ratio),
            'debt_service_ratio': debt_service_ratio,
            'credit_utilization': credit_data['credit_utilization'],
            'payment_history': credit_data['payment_history'],
            'good_credit': 1 if credit_data['credit_score'] >= 700 else 0,
            'debt_ratio_eligible': 1 if bank_data.get('debt_to_income_ratio', 0) <= 0.6 else 0,

            # Employment features
            'employed': 1 if bank_data['employment_status'] == 'Employed' else 0,
            'unemployed': 1 if bank_data['employment_status'] == 'Unemployed' else 0,
            'self_employed': 1 if bank_data['employment_status'] == 'Self-employed' else 0,
            'part_time': 1 if bank_data['employment_status'] == 'Part-time' else 0,

            # Experience features
            'total_experience_years': experience_years,
            'experience_ratio': experience_ratio,
            'has_experience': 1 if experience_years > 0 else 0,
            'experienced_professional': 1 if experience_years >= 5 else 0,

            # Career and skills
            'career_field': resume_data.get('career_field', 'unknown'),
            'num_skills': len(resume_data.get('skills', [])),
            'num_certifications': len(resume_data.get('certifications', [])),
            'num_languages': len(resume_data.get('languages', [])),
            'tech_career': 1 if resume_data.get('career_field') == 'technology' else 0,

            # Family and social factors
            'married': 1 if applicant['marital_status'] == 'Married' else 0,
            'single_parent': 1 if (applicant['marital_status'] in ['Divorced', 'Widowed'] and applicant['dependents'] > 0) else 0,
            'large_family': 1 if family_size >= 5 else 0,

            # Support type
            'support_financial': 1 if app['support_type'] in ['financial', 'both'] else 0,
            'support_economic': 1 if app['support_type'] in ['economic_enablement', 'both'] else 0,

            # Financial stress indicators
            'negative_cash_flow': 1 if monthly_income < monthly_expenses else 0,
            'high_debt_burden': 1 if debt_service_ratio > 0.4 else 0,
            'low_savings': 1 if savings_rate < 0.1 else 0,
            'financial_stress_score': self._calculate_financial_stress(monthly_income, monthly_expenses, debt_service_ratio, net_worth, family_size),

            # Age eligibility
            'age_eligible': 1 if 18 <= age <= 65 else 0,
            'young_adult': 1 if 18 <= age <= 30 else 0,
            'middle_aged': 1 if 30 < age <= 50 else 0,
            'senior': 1 if age > 50 else 0,
        }

        return features

    def _encode_education(self, education_level: str) -> int:
        """Encode education level as ordinal variable"""
        education_mapping = {
            'High School': 1,
            'Vocational Training': 2,
            'Technical Certificate': 3,
            'Diploma': 4,
            'Bachelor\'s Degree': 5,
            'Master\'s Degree': 6,
            'PhD': 7
        }
        return education_mapping.get(education_level, 1)

    def _calculate_financial_stress(self, income: float, expenses: float, debt_ratio: float, net_worth: float, family_size: int) -> float:
        """Calculate composite financial stress score"""
        stress_score = 0

        # Income adequacy
        per_capita_income = income / family_size
        if per_capita_income < 2000:  # Below poverty line
            stress_score += 0.4
        elif per_capita_income < 4000:  # Low income
            stress_score += 0.2

        # Cash flow stress
        if income < expenses:
            stress_score += 0.3
        elif (income - expenses) / income < 0.1:  # Less than 10% savings
            stress_score += 0.1

        # Debt burden
        if debt_ratio > 0.6:
            stress_score += 0.2
        elif debt_ratio > 0.4:
            stress_score += 0.1

        # Asset cushion
        if net_worth < 10000:
            stress_score += 0.1

        return min(stress_score, 1.0)  # Cap at 1.0

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare features and target variables for training"""
        logger.info("Preparing features for training...")

        # Separate features and targets
        feature_columns = [col for col in df.columns if col not in ['eligibility_score', 'expected_decision', 'career_field']]

        X = df[feature_columns].copy()
        y_score = df['eligibility_score']  # Regression target
        y_decision = df['expected_decision']  # Classification target

        # Handle categorical variables
        if 'career_field' in df.columns:
            # One-hot encode career field
            career_dummies = pd.get_dummies(df['career_field'], prefix='career')
            X = pd.concat([X, career_dummies], axis=1)

        # Fill any missing values
        X = X.fillna(0)

        # Split into train and test sets
        X_train, X_test, y_score_train, y_score_test = train_test_split(
            X, y_score, test_size=0.2, random_state=42, stratify=y_decision
        )

        # Also split decision targets
        _, _, y_decision_train, y_decision_test = train_test_split(
            X, y_decision, test_size=0.2, random_state=42, stratify=y_decision
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Target distribution: {y_decision.value_counts().to_dict()}")

        return X_train, X_test, y_score_train, y_score_test, y_decision_train, y_decision_test

    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple models and select the best one"""
        logger.info("Training ML models...")

        # Prepare data
        X_train, X_test, y_score_train, y_score_test, y_decision_train, y_decision_test = self.prepare_features(df)

        # Store feature names for later use
        self.feature_names = X_train.columns.tolist()

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['eligibility'] = scaler

        results = {}

        # 1. Random Forest for decision classification
        logger.info("Training Random Forest classifier...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }

        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_macro', n_jobs=-1)
        rf_grid.fit(X_train, y_decision_train)

        best_rf = rf_grid.best_estimator_
        rf_pred = best_rf.predict(X_test)
        rf_accuracy = accuracy_score(y_decision_test, rf_pred)

        self.models['random_forest'] = best_rf
        self.feature_importance['random_forest'] = dict(zip(self.feature_names, best_rf.feature_importances_))
        results['random_forest'] = {
            'accuracy': rf_accuracy,
            'classification_report': classification_report(y_decision_test, rf_pred, output_dict=True),
            'best_params': rf_grid.best_params_
        }

        logger.info(f"Random Forest accuracy: {rf_accuracy:.3f}")

        # 2. Gradient Boosting for decision classification
        logger.info("Training Gradient Boosting classifier...")
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7]
        }

        gb = GradientBoostingClassifier(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='f1_macro', n_jobs=-1)
        gb_grid.fit(X_train, y_decision_train)

        best_gb = gb_grid.best_estimator_
        gb_pred = best_gb.predict(X_test)
        gb_accuracy = accuracy_score(y_decision_test, gb_pred)

        self.models['gradient_boosting'] = best_gb
        self.feature_importance['gradient_boosting'] = dict(zip(self.feature_names, best_gb.feature_importances_))

        results['gradient_boosting'] = {
            'accuracy': gb_accuracy,
            'classification_report': classification_report(y_decision_test, gb_pred, output_dict=True),
            'best_params': gb_grid.best_params_
        }

        logger.info(f"Gradient Boosting accuracy: {gb_accuracy:.3f}")

        # 3. Logistic Regression with scaled features
        logger.info("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train_scaled, y_decision_train)

        lr_pred = lr.predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_decision_test, lr_pred)

        self.models['logistic_regression'] = lr

        results['logistic_regression'] = {
            'accuracy': lr_accuracy,
            'classification_report': classification_report(y_decision_test, lr_pred, output_dict=True)
        }

        logger.info(f"Logistic Regression accuracy: {lr_accuracy:.3f}")

        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name

        logger.info(f"Best model: {best_model_name} (accuracy: {results[best_model_name]['accuracy']:.3f})")

        # Store test data for final evaluation
        self.test_data = {
            'X_test': X_test,
            'y_decision_test': y_decision_test,
            'y_score_test': y_score_test
        }

        return results

    def save_models(self, output_path: str = "models"):
        """Save trained models and preprocessing objects"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        # Save models
        for name, model in self.models.items():
            joblib.dump(model, output_dir / f"{name}_model.pkl")

        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, output_dir / f"{name}_scaler.pkl")

        # Save feature names and importance
        model_metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'best_model': self.best_model_name,
            'training_date': datetime.now().isoformat()
        }

        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        logger.info(f"Models saved to {output_dir}")

    def evaluate_model(self) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        if not hasattr(self, 'test_data'):
            raise ValueError("No test data available. Run train_models first.")

        X_test = self.test_data['X_test']
        y_test = self.test_data['y_decision_test']

        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)

        # Get feature importance for the best model
        top_features = sorted(
            self.feature_importance[self.best_model_name].items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        evaluation = {
            'model_name': self.best_model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'top_features': top_features,
            'feature_count': len(self.feature_names)
        }

        return evaluation

def main():
    """Main training pipeline"""
    # Initialize trainer
    trainer = EligibilityModelTrainer()

    # Load and prepare data
    df = trainer.load_training_data()

    # Train models
    results = trainer.train_models(df)

    # Save models
    trainer.save_models()

    # Evaluate best model
    evaluation = trainer.evaluate_model()

    # Print results
    print(f"\n🏆 Best Model: {evaluation['model_name']}")
    print(f"📊 Accuracy: {evaluation['accuracy']:.3f}")
    print(f"🔍 Features used: {evaluation['feature_count']}")
    print(f"\n🔥 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(evaluation['top_features'][:10], 1):
        print(f"   {i:2d}. {feature}: {importance:.3f}")

    return trainer, results, evaluation

if __name__ == "__main__":
    trainer, results, evaluation = main()