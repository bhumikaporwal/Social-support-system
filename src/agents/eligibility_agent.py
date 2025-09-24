from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from .base_agent import BaseAgent, AgentType, AgentResult

class EligibilityAssessmentAgent(BaseAgent):
    """Agent responsible for assessing applicant eligibility for social support"""

    def __init__(self, llm_client=None, langfuse_client=None):
        super().__init__(AgentType.ELIGIBILITY_ASSESSMENT, llm_client, langfuse_client)
        self.eligibility_model = self._initialize_eligibility_model()
        self.scaler = StandardScaler()
        self.eligibility_criteria = self._load_eligibility_criteria()

    def _initialize_eligibility_model(self) -> RandomForestClassifier:
        """Initialize the ML model for eligibility assessment"""
        # In production, this would load a pre-trained model
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
            ml_score = self._calculate_ml_score(features)

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

    def _calculate_ml_score(self, features: Dict[str, Any]) -> float:
        """Calculate ML-based eligibility score"""
        try:
            # Create feature vector for ML model
            feature_vector = [
                features['age'],
                features['monthly_income'],
                features['net_worth'],
                features['debt_to_income_ratio'],
                features['credit_score'],
                features['experience_years'],
                1 if features['has_high_demand_skills'] else 0,
                features['family_size'],
                features['dependents'],
                features['document_confidence']
            ]

            # In production, use trained model
            # For demo, use a simplified scoring function
            score = self._simplified_ml_scoring(feature_vector)

            return min(max(score, 0.0), 1.0)

        except Exception as e:
            self.logger.error(f"ML scoring failed: {str(e)}")
            return 0.5  # Default score

    def _simplified_ml_scoring(self, features: List[float]) -> float:
        """Simplified ML scoring for demo purposes"""
        # Normalize and weight features
        weights = [0.1, 0.3, 0.2, -0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1]

        # Simple weighted scoring
        normalized_features = []
        for i, (feature, weight) in enumerate(zip(features, weights)):
            if i == 0:  # Age
                normalized = min(feature / 50, 1.0)
            elif i in [1, 2]:  # Income, net worth
                normalized = min(feature / 50000, 1.0) if feature > 0 else 0
            elif i == 3:  # Debt to income (negative weight)
                normalized = feature
            elif i == 4:  # Credit score
                normalized = feature / 850
            else:
                normalized = min(feature, 1.0)

            normalized_features.append(normalized * weight)

        return sum(normalized_features) / sum(abs(w) for w in weights)

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