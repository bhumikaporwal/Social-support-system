from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import json
from pathlib import Path
from .base_agent import BaseAgent, AgentType, AgentResult

class EligibilityAssessmentAgent(BaseAgent):
    """Agent responsible for assessing applicant eligibility for social support"""

    def __init__(self, llm_client=None, langfuse_client=None):
        super().__init__(AgentType.ELIGIBILITY_ASSESSMENT, llm_client, langfuse_client)
        self.eligibility_model = self._load_trained_model()
        self.scaler = self._load_scaler()
        self.feature_names = self._load_feature_names()
        self.eligibility_criteria = self._load_eligibility_criteria()
        self.model_loaded = self.eligibility_model is not None

    def _load_trained_model(self) -> Optional[RandomForestClassifier]:
        """Load the trained ML model for eligibility assessment"""
        try:
            model_path = Path("models")
            # Try to load the best model first
            metadata_file = model_path / "model_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                best_model_name = metadata.get('best_model', 'random_forest')
                model_file = model_path / f"{best_model_name}_model.pkl"
            else:
                model_file = model_path / "random_forest_model.pkl"

            if model_file.exists():
                model = joblib.load(model_file)
                self.logger.info(f"Loaded trained model from {model_file}")
                return model
            else:
                self.logger.warning(f"No trained model found at {model_file}")
                return self._create_fallback_model()
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {e}")
            return self._create_fallback_model()

    def _load_scaler(self) -> Optional[StandardScaler]:
        """Load the trained scaler"""
        try:
            scaler_path = Path("models") / "eligibility_scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                self.logger.info("Loaded trained scaler")
                return scaler
            else:
                self.logger.warning("No trained scaler found")
                return StandardScaler()
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {e}")
            return StandardScaler()

    def _load_feature_names(self) -> List[str]:
        """Load feature names from model metadata"""
        try:
            metadata_file = Path("models") / "model_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('feature_names', [])
        except Exception as e:
            self.logger.error(f"Failed to load feature names: {e}")
        return []

    def _create_fallback_model(self) -> RandomForestClassifier:
        """Create a basic model as fallback"""
        self.logger.warning("Using fallback model - predictions may not be accurate")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        return model

    def _load_eligibility_criteria(self) -> Dict[str, Any]:
        """Load eligibility criteria for social support"""
        return {
            "financial_support": {
                "max_monthly_income": 15000,  # AED
                "max_net_worth": 500000,      # AED
                "min_age": 18,
                "max_age": 65,
                "uae_resident": True,
                "debt_to_income_ratio": 0.6   # Max 60%
            },
            "economic_enablement": {
                "max_monthly_income": 25000,  # AED
                "min_age": 18,
                "max_age": 55,
                "uae_resident": True,
                "employment_status": ["unemployed", "underemployed", "seeking_upgrade"]
            }
        }

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Assess applicant eligibility for social support programs"""
        try:
            validated_data = input_data.get('validated_data', {})
            support_type = input_data.get('support_type', 'both')

            # Extract relevant features
            features = self._extract_features(validated_data)

            # Rule-based assessment
            rule_based_result = self._rule_based_assessment(features, support_type)

            # ML-based scoring (if model is trained)
            ml_result = self._calculate_ml_score(features)
            ml_score = ml_result.get('ml_score', 0.5)

            # Risk assessment
            risk_assessment = self._assess_risk_factors(features)

            # Generate LLM-based reasoning
            llm_reasoning = await self._generate_llm_reasoning(features, rule_based_result, risk_assessment)

            # Combine assessments
            final_assessment = self._combine_assessments(rule_based_result, ml_score, risk_assessment)

            return AgentResult(
                success=True,
                data={
                    "eligibility_assessment": final_assessment,
                    "rule_based_result": rule_based_result,
                    "ml_result": ml_result,
                    "ml_score": ml_score,
                    "risk_assessment": risk_assessment,
                    "features_used": features
                },
                reasoning=llm_reasoning,
                confidence=final_assessment.get('confidence_score', 0.5)
            )

        except Exception as e:
            self.logger.error(f"Eligibility assessment failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                reasoning=f"Eligibility assessment failed: {str(e)}",
                confidence=0.0,
                errors=[str(e)]
            )

    def _extract_features(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features for eligibility assessment"""
        features = {
            # Demographics
            'age': self._calculate_age(validated_data.get('date_of_birth')),
            'gender': validated_data.get('gender'),
            'nationality': validated_data.get('nationality'),
            'emirate': validated_data.get('emirate'),
            'marital_status': validated_data.get('marital_status'),

            # Financial
            'monthly_income': validated_data.get('monthly_income', 0),
            'net_worth': validated_data.get('net_worth', 0),
            'debt_to_income_ratio': validated_data.get('debt_to_income_ratio', 0),
            'credit_score': validated_data.get('credit_score', 650),
            'liquid_assets': validated_data.get('liquid_assets', 0),
            'total_liabilities': validated_data.get('total_liabilities', 0),

            # Employment
            'employment_status': validated_data.get('employment_status'),
            'experience_years': validated_data.get('experience_years', 0),
            'has_high_demand_skills': validated_data.get('has_high_demand_skills', False),
            'education_level': validated_data.get('education_level'),

            # Family
            'family_size': validated_data.get('family_size', 1),
            'dependents': validated_data.get('dependents', 0),

            # Document quality
            'document_confidence': validated_data.get('average_document_confidence', 0.5)
        }

        return features

    def _calculate_age(self, date_of_birth) -> int:
        """Calculate age from date of birth"""
        if not date_of_birth:
            return 0

        try:
            from datetime import date
            if isinstance(date_of_birth, str):
                from datetime import datetime
                dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            else:
                dob = date_of_birth

            today = date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except:
            return 0

    def _rule_based_assessment(self, features: Dict[str, Any], support_type: str) -> Dict[str, Any]:
        """Perform rule-based eligibility assessment"""
        assessment = {
            'financial_support_eligible': False,
            'economic_enablement_eligible': False,
            'eligibility_reasons': [],
            'disqualification_reasons': []
        }

        criteria = self.eligibility_criteria

        # Financial Support Assessment
        if support_type in ['financial', 'both']:
            financial_eligible = True
            reasons = []

            # Income check
            if features['monthly_income'] > criteria['financial_support']['max_monthly_income']:
                financial_eligible = False
                assessment['disqualification_reasons'].append(
                    f"Monthly income ({features['monthly_income']} AED) exceeds limit ({criteria['financial_support']['max_monthly_income']} AED)"
                )
            else:
                reasons.append("Income within eligible range")

            # Net worth check
            if features['net_worth'] > criteria['financial_support']['max_net_worth']:
                financial_eligible = False
                assessment['disqualification_reasons'].append(
                    f"Net worth ({features['net_worth']} AED) exceeds limit ({criteria['financial_support']['max_net_worth']} AED)"
                )
            else:
                reasons.append("Net worth within eligible range")

            # Age check
            age = features['age']
            if age < criteria['financial_support']['min_age'] or age > criteria['financial_support']['max_age']:
                financial_eligible = False
                assessment['disqualification_reasons'].append(
                    f"Age ({age}) outside eligible range ({criteria['financial_support']['min_age']}-{criteria['financial_support']['max_age']})"
                )
            else:
                reasons.append("Age within eligible range")

            # Debt-to-income ratio
            if features['debt_to_income_ratio'] > criteria['financial_support']['debt_to_income_ratio']:
                assessment['disqualification_reasons'].append(
                    f"High debt-to-income ratio ({features['debt_to_income_ratio']:.2%})"
                )
                # This is a warning, not a disqualifier
            else:
                reasons.append("Debt-to-income ratio acceptable")

            assessment['financial_support_eligible'] = financial_eligible
            if financial_eligible:
                assessment['eligibility_reasons'].extend(reasons)

        # Economic Enablement Assessment
        if support_type in ['economic_enablement', 'both']:
            economic_eligible = True
            reasons = []

            # Income check (higher threshold)
            if features['monthly_income'] > criteria['economic_enablement']['max_monthly_income']:
                economic_eligible = False
                assessment['disqualification_reasons'].append(
                    f"Income too high for economic enablement ({features['monthly_income']} AED)"
                )
            else:
                reasons.append("Income suitable for economic enablement")

            # Age check
            age = features['age']
            if age < criteria['economic_enablement']['min_age'] or age > criteria['economic_enablement']['max_age']:
                economic_eligible = False
                assessment['disqualification_reasons'].append(
                    f"Age ({age}) outside economic enablement range ({criteria['economic_enablement']['min_age']}-{criteria['economic_enablement']['max_age']})"
                )
            else:
                reasons.append("Age suitable for economic enablement")

            # Skills and experience assessment
            if features['has_high_demand_skills']:
                reasons.append("Has high-demand skills suitable for training")

            if features['experience_years'] < 10:
                reasons.append("Has room for career growth and development")

            assessment['economic_enablement_eligible'] = economic_eligible
            if economic_eligible:
                assessment['eligibility_reasons'].extend(reasons)

        return assessment

    def _calculate_ml_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ML-based eligibility score using trained model"""
        try:
            if not self.model_loaded:
                self.logger.warning("No trained model available, using fallback scoring")
                return self._fallback_ml_scoring(features)

            # Create feature vector matching training data format
            ml_features = self._create_ml_feature_vector(features)

            if not ml_features:
                return self._fallback_ml_scoring(features)

            # Convert to DataFrame with proper feature names
            feature_df = pd.DataFrame([ml_features], columns=self.feature_names)

            # Get predictions
            prediction = self.eligibility_model.predict(feature_df)[0]
            prediction_proba = self.eligibility_model.predict_proba(feature_df)[0]

            # Get confidence score (max probability)
            confidence = max(prediction_proba)

            # Map decision to score
            decision_to_score = {
                'approve': 0.9,
                'conditional_approve': 0.7,
                'manual_review': 0.5,
                'decline': 0.2
            }

            ml_score = decision_to_score.get(prediction, 0.5)

            return {
                'ml_score': ml_score,
                'ml_prediction': prediction,
                'ml_confidence': confidence,
                'prediction_probabilities': dict(zip(self.eligibility_model.classes_, prediction_proba))
            }

        except Exception as e:
            self.logger.error(f"ML scoring failed: {str(e)}")
            return self._fallback_ml_scoring(features)

    def _create_ml_feature_vector(self, features: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Create feature vector matching the trained model format"""
        try:
            # Extract all the features that were used during training
            age = features.get('age', 30)
            monthly_income = features.get('monthly_income', 0)
            net_worth = features.get('net_worth', 0)
            family_size = features.get('family_size', 1)
            dependents = features.get('dependents', 0)
            credit_score = features.get('credit_score', 650)
            debt_ratio = features.get('debt_to_income_ratio', 0.3)
            experience_years = features.get('experience_years', 0)

            # Calculate derived features to match training data
            per_capita_income = monthly_income / family_size if family_size > 0 else monthly_income

            # Create feature vector matching training format
            ml_features = {
                'age': age,
                'gender_male': 1 if features.get('gender') == 'Male' else 0,
                'family_size': family_size,
                'dependents': dependents,
                'emirate_dubai': 1 if features.get('emirate') == 'Dubai' else 0,
                'emirate_abu_dhabi': 1 if features.get('emirate') == 'Abu Dhabi' else 0,
                'high_cost_emirate': 1 if features.get('emirate') in ['Dubai', 'Abu Dhabi'] else 0,
                'uae_national': 1 if features.get('nationality') == 'UAE' else 0,
                'education_level': self._encode_education(features.get('education_level', 'High School')),
                'has_degree': 1 if 'Degree' in str(features.get('education_level', '')) else 0,
                'has_advanced_degree': 1 if features.get('education_level') in ['Master\'s Degree', 'PhD'] else 0,
                'monthly_income': monthly_income,
                'monthly_expenses': monthly_income * 0.8,  # Estimate if not available
                'monthly_net': monthly_income * 0.2,
                'per_capita_income': per_capita_income,
                'savings_rate': 0.2,  # Default estimate
                'income_eligible_financial': 1 if monthly_income <= 15000 else 0,
                'income_eligible_economic': 1 if monthly_income <= 25000 else 0,
                'very_low_income': 1 if monthly_income <= 5000 else 0,
                'low_income': 1 if 5000 < monthly_income <= 10000 else 0,
                'medium_income': 1 if 10000 < monthly_income <= 20000 else 0,
                'net_worth': net_worth,
                'total_assets': net_worth + 50000,  # Estimate
                'total_liabilities': max(0, 50000 - net_worth),
                'liquid_assets': net_worth * 0.3,
                'asset_liquidity_ratio': 0.3,
                'net_worth_eligible': 1 if net_worth <= 500000 else 0,
                'credit_score': credit_score,
                'debt_to_income_ratio': debt_ratio,
                'debt_service_ratio': debt_ratio,
                'credit_utilization': min(debt_ratio * 100, 85),
                'payment_history': max(70, 100 - debt_ratio * 30),
                'good_credit': 1 if credit_score >= 700 else 0,
                'debt_ratio_eligible': 1 if debt_ratio <= 0.6 else 0,
                'employed': 1 if features.get('employment_status') == 'Employed' else 0,
                'unemployed': 1 if features.get('employment_status') == 'Unemployed' else 0,
                'self_employed': 1 if features.get('employment_status') == 'Self-employed' else 0,
                'part_time': 1 if features.get('employment_status') == 'Part-time' else 0,
                'total_experience_years': experience_years,
                'experience_ratio': experience_years / (age - 18) if age > 18 else 0,
                'has_experience': 1 if experience_years > 0 else 0,
                'experienced_professional': 1 if experience_years >= 5 else 0,
                'num_skills': 5,  # Default estimate
                'num_certifications': 1,  # Default estimate
                'num_languages': 2,  # Default estimate
                'tech_career': 1 if features.get('has_high_demand_skills') else 0,
                'married': 1 if features.get('marital_status') == 'Married' else 0,
                'single_parent': 1 if (features.get('marital_status') in ['Divorced', 'Widowed'] and dependents > 0) else 0,
                'large_family': 1 if family_size >= 5 else 0,
                'support_financial': 1,  # Default
                'support_economic': 1,   # Default
                'negative_cash_flow': 1 if monthly_income < (monthly_income * 0.8) else 0,
                'high_debt_burden': 1 if debt_ratio > 0.4 else 0,
                'low_savings': 1 if (monthly_income * 0.2) / monthly_income < 0.1 else 0,
                'financial_stress_score': min(debt_ratio + (1 - min(per_capita_income/5000, 1)), 1.0),
                'age_eligible': 1 if 18 <= age <= 65 else 0,
                'young_adult': 1 if 18 <= age <= 30 else 0,
                'middle_aged': 1 if 30 < age <= 50 else 0,
                'senior': 1 if age > 50 else 0
            }

            # Add any missing features with default values
            if self.feature_names:
                for feature_name in self.feature_names:
                    if feature_name not in ml_features:
                        if 'career_' in feature_name:
                            ml_features[feature_name] = 0  # Career field one-hot encoding
                        else:
                            ml_features[feature_name] = 0  # Default for missing features

            return ml_features

        except Exception as e:
            self.logger.error(f"Failed to create ML feature vector: {e}")
            return None

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

    def _fallback_ml_scoring(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback ML scoring when trained model is not available"""
        # Simplified scoring based on key eligibility factors
        score = 0.5  # Base score

        age = features.get('age', 30)
        monthly_income = features.get('monthly_income', 0)
        net_worth = features.get('net_worth', 0)
        debt_ratio = features.get('debt_to_income_ratio', 0.3)

        # Age factor
        if 18 <= age <= 65:
            score += 0.1

        # Income factor
        if monthly_income <= 15000:
            score += 0.2
            if monthly_income < 5000:
                score += 0.1

        # Wealth factor
        if net_worth <= 500000:
            score += 0.1

        # Debt factor
        if debt_ratio <= 0.6:
            score += 0.1

        score = min(max(score, 0.0), 1.0)

        # Map to decision
        if score >= 0.8:
            prediction = 'approve'
        elif score >= 0.6:
            prediction = 'conditional_approve'
        elif score >= 0.4:
            prediction = 'manual_review'
        else:
            prediction = 'decline'

        return {
            'ml_score': score,
            'ml_prediction': prediction,
            'ml_confidence': 0.6,  # Lower confidence for fallback
            'prediction_probabilities': {prediction: 0.6, 'manual_review': 0.4}
        }


    def _assess_risk_factors(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk factors for the application"""
        risk_factors = []
        risk_score = 0.0

        # Financial risk factors
        if features['debt_to_income_ratio'] > 0.5:
            risk_factors.append("High debt-to-income ratio")
            risk_score += 0.3

        if features['credit_score'] < 600:
            risk_factors.append("Poor credit score")
            risk_score += 0.2

        if features['monthly_income'] < 3000:
            risk_factors.append("Very low income")
            risk_score += 0.2

        # Employment risk factors
        if features['experience_years'] < 1:
            risk_factors.append("Limited work experience")
            risk_score += 0.1

        # Document quality risk
        if features['document_confidence'] < 0.6:
            risk_factors.append("Low document confidence")
            risk_score += 0.2

        return {
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'Low'
        }

    async def _generate_llm_reasoning(self, features: Dict[str, Any], rule_assessment: Dict[str, Any], risk_assessment: Dict[str, Any]) -> str:
        """Generate LLM-based reasoning for the assessment"""
        try:
            prompt_template = """
            Based on the following applicant information, provide a comprehensive eligibility assessment reasoning:

            Applicant Features:
            - Age: {age}
            - Monthly Income: {monthly_income} AED
            - Net Worth: {net_worth} AED
            - Credit Score: {credit_score}
            - Family Size: {family_size}
            - Employment Experience: {experience_years} years
            - Has High-Demand Skills: {has_high_demand_skills}

            Rule-Based Assessment:
            - Financial Support Eligible: {financial_eligible}
            - Economic Enablement Eligible: {economic_eligible}
            - Eligibility Reasons: {eligibility_reasons}
            - Disqualification Reasons: {disqualification_reasons}

            Risk Assessment:
            - Risk Level: {risk_level}
            - Risk Factors: {risk_factors}

            Provide a clear, professional assessment explaining the eligibility decision and any recommendations.
            """

            prompt = self._create_prompt(prompt_template, {
                'age': features['age'],
                'monthly_income': features['monthly_income'],
                'net_worth': features['net_worth'],
                'credit_score': features['credit_score'],
                'family_size': features['family_size'],
                'experience_years': features['experience_years'],
                'has_high_demand_skills': features['has_high_demand_skills'],
                'financial_eligible': rule_assessment['financial_support_eligible'],
                'economic_eligible': rule_assessment['economic_enablement_eligible'],
                'eligibility_reasons': rule_assessment['eligibility_reasons'],
                'disqualification_reasons': rule_assessment['disqualification_reasons'],
                'risk_level': risk_assessment['risk_level'],
                'risk_factors': risk_assessment['risk_factors']
            })

            system_message = "You are an expert social support eligibility assessor. Provide clear, fair, and comprehensive assessments."

            reasoning = await self._call_llm(prompt, system_message)
            return reasoning

        except Exception as e:
            self.logger.error(f"LLM reasoning generation failed: {str(e)}")
            return "Automated assessment completed. Please review the detailed criteria matching results."

    def _combine_assessments(self, rule_result: Dict[str, Any], ml_score: float, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Combine different assessment methods into final result"""
        # Weight the different assessment methods
        rule_weight = 0.6
        ml_weight = 0.3
        risk_weight = 0.1

        # Calculate combined score
        rule_score = 0.8 if rule_result['financial_support_eligible'] or rule_result['economic_enablement_eligible'] else 0.2
        risk_penalty = risk_assessment['risk_score'] * risk_weight

        combined_score = (rule_score * rule_weight) + (ml_score * ml_weight) - risk_penalty

        # Determine final recommendation
        if combined_score > 0.7 and (rule_result['financial_support_eligible'] or rule_result['economic_enablement_eligible']):
            recommendation = "approve"
        elif combined_score > 0.4:
            recommendation = "conditional_approve"
        else:
            recommendation = "decline"

        return {
            'combined_score': min(max(combined_score, 0.0), 1.0),
            'recommendation': recommendation,
            'confidence_score': combined_score,
            'rule_based_eligible': rule_result['financial_support_eligible'] or rule_result['economic_enablement_eligible'],
            'ml_score': ml_score,
            'risk_level': risk_assessment['risk_level']
        }