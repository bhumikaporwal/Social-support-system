from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentType, AgentResult
from datetime import datetime

class DecisionRecommendationAgent(BaseAgent):
    """Agent responsible for making final approval/decline recommendations"""

    def __init__(self, llm_client=None, langfuse_client=None):
        super().__init__(AgentType.DECISION_RECOMMENDATION, llm_client, langfuse_client)
        self.decision_thresholds = self._load_decision_thresholds()
        self.approval_limits = self._load_approval_limits()

    def _load_decision_thresholds(self) -> Dict[str, float]:
        """Load decision thresholds for different recommendation levels"""
        return {
            'auto_approve': 0.85,
            'conditional_approve': 0.65,
            'manual_review': 0.45,
            'auto_decline': 0.30
        }

    def _load_approval_limits(self) -> Dict[str, Dict[str, Any]]:
        """Load approval limits for different support types"""
        return {
            'financial_support': {
                'monthly_amount_range': [500, 5000],  # AED
                'duration_months_range': [3, 12],
                'max_total_amount': 50000
            },
            'economic_enablement': {
                'training_budget': 15000,
                'duration_months': 6,
                'job_placement_support': True
            }
        }

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Make final decision recommendation based on all assessments"""
        try:
            eligibility_data = input_data.get('eligibility_assessment', {})
            validation_data = input_data.get('validation_result', {})
            extracted_data = input_data.get('extracted_documents', {})
            support_type = input_data.get('support_type', 'both')

            # Analyze all available data
            decision_factors = self._analyze_decision_factors(
                eligibility_data, validation_data, extracted_data, support_type
            )

            # Make recommendation
            recommendation = self._make_recommendation(decision_factors)

            # Calculate support amounts if approved
            support_details = self._calculate_support_details(
                recommendation, decision_factors, support_type
            )

            # Generate comprehensive reasoning
            reasoning = await self._generate_decision_reasoning(
                recommendation, decision_factors, support_details
            )

            # Create audit trail
            audit_trail = self._create_audit_trail(
                recommendation, decision_factors, support_details
            )

            return AgentResult(
                success=True,
                data={
                    "final_recommendation": recommendation,
                    "support_details": support_details,
                    "decision_factors": decision_factors,
                    "audit_trail": audit_trail
                },
                reasoning=reasoning,
                confidence=recommendation.get('confidence', 0.5)
            )

        except Exception as e:
            self.logger.error(f"Decision recommendation failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                reasoning=f"Decision recommendation failed: {str(e)}",
                confidence=0.0,
                errors=[str(e)]
            )

    def _analyze_decision_factors(self, eligibility_data: Dict[str, Any],
                                 validation_data: Dict[str, Any],
                                 extracted_data: Dict[str, Any],
                                 support_type: str) -> Dict[str, Any]:
        """Analyze all factors that influence the decision"""
        validated_data = validation_data.get('validated_data', {})

        factors = {
            'eligibility_score': eligibility_data.get('combined_score', 0.5),
            'rule_based_eligible': eligibility_data.get('rule_based_eligible', False),
            'ml_score': eligibility_data.get('ml_score', 0.5),
            'risk_level': eligibility_data.get('risk_level', 'Medium'),
            'validation_confidence': validation_data.get('confidence_score', 0.5),
            'validation_issues': len(validation_data.get('issues', [])),
            'critical_validation_errors': len([
                issue for issue in validation_data.get('issues', [])
                if issue.get('severity') == 'error'
            ]),
            'document_quality': self._assess_document_quality(extracted_data),
            'financial_need_level': self._assess_financial_need(validated_data),
            'validated_data': validated_data,  # Include validated data for calculations
            'validation_result': validation_data,  # Include full validation result
            'support_type': support_type
        }

        return factors

    def _assess_document_quality(self, extracted_data: Dict[str, Any]) -> float:
        """Assess overall document quality"""
        if not extracted_data:
            return 0.0

        quality_scores = []
        required_docs = ['emirates_id', 'bank_statement']

        for doc_type, data in extracted_data.items():
            if data:
                filled_fields = sum(1 for v in data.values() if v is not None and v != "")
                total_fields = len(data)
                doc_quality = filled_fields / total_fields if total_fields > 0 else 0
                quality_scores.append(doc_quality)

        # Penalty for missing required documents
        missing_required = len([doc for doc in required_docs if doc not in extracted_data])
        penalty = missing_required * 0.2

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        return max(0, avg_quality - penalty)

    def _assess_financial_need(self, validated_data: Dict[str, Any]) -> str:
        """Assess the level of financial need"""
        monthly_income = validated_data.get('monthly_income', 0)
        family_size = validated_data.get('family_size', 1)
        dependents = validated_data.get('dependents', 0)
        net_worth = validated_data.get('net_worth', 0)

        # Calculate per-capita income
        per_capita_income = monthly_income / family_size if family_size > 0 else monthly_income

        # UAE poverty line approximation
        poverty_threshold = 2000  # AED per person per month
        low_income_threshold = 4000

        if per_capita_income < poverty_threshold:
            return "critical"
        elif per_capita_income < low_income_threshold:
            return "high"
        elif net_worth < 50000 and dependents > 0:
            return "moderate"
        else:
            return "low"

    def _make_recommendation(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Make the final recommendation based on decision factors"""
        eligibility_score = factors['eligibility_score']
        rule_based_eligible = factors['rule_based_eligible']
        critical_errors = factors['critical_validation_errors']
        document_quality = factors['document_quality']
        validation_confidence = factors['validation_confidence']
        risk_level = factors['risk_level']
        validated_data = factors.get('validated_data', {})

        # Critical rejection scenarios
        monthly_income = validated_data.get('monthly_income', 0)
        net_worth = validated_data.get('net_worth', 0)
        age = validated_data.get('age', 0)
        debt_ratio = validated_data.get('debt_to_income_ratio', 0)

        # Cannot approve if critical validation errors exist
        if critical_errors > 0:
            return {
                'decision': 'decline',
                'reason': 'Critical validation errors in submitted documents',
                'confidence': 0.9,
                'requires_manual_review': False
            }

        # Age eligibility check
        if age < 18 or age > 65:
            return {
                'decision': 'decline',
                'reason': f'Age ({age}) outside eligible range (18-65 years)',
                'confidence': 0.95,
                'requires_manual_review': False
            }

        # Income too high for financial support
        if monthly_income > 25000:
            return {
                'decision': 'decline',
                'reason': f'Monthly income ({monthly_income:,} AED) exceeds maximum threshold (25,000 AED)',
                'confidence': 0.9,
                'requires_manual_review': False
            }

        # Net worth too high
        if net_worth > 500000:
            return {
                'decision': 'decline',
                'reason': f'Net worth ({net_worth:,} AED) exceeds maximum threshold (500,000 AED)',
                'confidence': 0.9,
                'requires_manual_review': False
            }

        # Excessive debt burden
        if debt_ratio > 0.8:
            return {
                'decision': 'decline',
                'reason': f'Debt-to-income ratio ({debt_ratio:.1%}) indicates unsustainable debt burden',
                'confidence': 0.85,
                'requires_manual_review': False
            }

        # Cannot approve without basic eligibility
        if not rule_based_eligible:
            return {
                'decision': 'decline',
                'reason': 'Does not meet basic eligibility criteria for requested support type',
                'confidence': 0.8,
                'requires_manual_review': False
            }

        # High risk with low income might need manual review
        if risk_level == 'High' and monthly_income < 3000:
            return {
                'decision': 'manual_review',
                'reason': 'High-risk profile with very low income requires manual assessment',
                'confidence': 0.6,
                'requires_manual_review': True
            }

        # Document quality check
        if document_quality < 0.4:
            return {
                'decision': 'decline',
                'reason': 'Document quality insufficient for verification',
                'confidence': 0.8,
                'requires_manual_review': False
            }
        elif document_quality < 0.6:
            return {
                'decision': 'manual_review',
                'reason': 'Poor document quality requires manual verification',
                'confidence': 0.7,
                'requires_manual_review': True
            }

        # Low validation confidence
        if validation_confidence < 0.5:
            return {
                'decision': 'decline',
                'reason': 'Data validation confidence too low for automated processing',
                'confidence': 0.75,
                'requires_manual_review': False
            }

        # Decision based on combined score and thresholds
        thresholds = self.decision_thresholds

        if eligibility_score >= thresholds['auto_approve'] and validation_confidence >= 0.8:
            return {
                'decision': 'approve',
                'reason': 'Meets all criteria for automatic approval',
                'confidence': eligibility_score,
                'requires_manual_review': False
            }
        elif eligibility_score >= thresholds['conditional_approve']:
            return {
                'decision': 'conditional_approve',
                'reason': 'Eligible with conditions or limitations',
                'confidence': eligibility_score,
                'requires_manual_review': False
            }
        elif eligibility_score >= thresholds['manual_review']:
            return {
                'decision': 'manual_review',
                'reason': 'Borderline case requiring manual assessment',
                'confidence': 0.6,
                'requires_manual_review': True
            }
        else:
            return {
                'decision': 'decline',
                'reason': 'Does not meet minimum eligibility threshold',
                'confidence': 0.8,
                'requires_manual_review': False
            }

    def _calculate_support_details(self, recommendation: Dict[str, Any],
                                  factors: Dict[str, Any],
                                  support_type: str) -> Dict[str, Any]:
        """Calculate specific support amounts and details if approved"""
        if recommendation['decision'] in ['decline']:
            return {
                'financial_support': None,
                'economic_enablement': None,
                'decline_reason': recommendation.get('reason', 'Application does not meet eligibility requirements')
            }

        details = {}
        limits = self.approval_limits

        need_level = factors.get('financial_need_level', 'moderate')
        eligibility_score = factors.get('eligibility_score', 0.5)

        # Get validated data for more precise calculations
        validated_data = factors.get('validated_data', {})
        if not validated_data:
            # Extract from input data if not in factors
            input_validation_data = factors.get('validation_result', {})
            validated_data = input_validation_data.get('validated_data', {})

        monthly_income = validated_data.get('monthly_income', 0)
        family_size = validated_data.get('family_size', 1)
        dependents = validated_data.get('dependents', 0)

        # Financial Support Calculation
        if support_type in ['financial', 'both']:
            financial_limits = limits['financial_support']

            # Calculate poverty gap and need-based amount
            per_capita_income = monthly_income / family_size if family_size > 0 else monthly_income
            poverty_line = 2000  # AED per person per month
            basic_living_cost = 3500  # AED per person per month

            # Calculate support based on income gap and family composition
            if need_level == 'critical':
                # Full gap coverage up to maximum
                income_gap = max(0, (basic_living_cost * family_size) - monthly_income)
                base_amount = min(income_gap, financial_limits['monthly_amount_range'][1])
            elif need_level == 'high':
                # 80% gap coverage
                income_gap = max(0, (poverty_line * family_size) - monthly_income) * 0.8
                base_amount = min(income_gap, financial_limits['monthly_amount_range'][1] * 0.8)
            elif need_level == 'moderate':
                # Fixed moderate amount based on family size
                base_amount = min(1500 + (dependents * 500), financial_limits['monthly_amount_range'][1] * 0.6)
            else:
                # Low need gets minimal support
                base_amount = financial_limits['monthly_amount_range'][0]

            # Adjust based on eligibility score
            score_multiplier = max(0.5, min(eligibility_score * 1.1, 1.0))
            monthly_amount = max(financial_limits['monthly_amount_range'][0], base_amount * score_multiplier)

            # Determine duration based on need and risk level
            if need_level == 'critical' and factors.get('risk_level') != 'High':
                duration = financial_limits['duration_months_range'][1]
            elif need_level in ['critical', 'high']:
                duration = max(6, financial_limits['duration_months_range'][1] - 2)
            else:
                duration = financial_limits['duration_months_range'][0]

            # Conditional approval adjustments
            if recommendation['decision'] == 'conditional_approve':
                monthly_amount *= 0.75  # Reduce amount for conditional approval
                duration = min(duration, 6)  # Limit duration

            # Manual review cases get reduced initial amounts
            elif recommendation['decision'] == 'manual_review':
                monthly_amount *= 0.6
                duration = 3  # Short initial period

            details['financial_support'] = {
                'monthly_amount': round(monthly_amount, 2),
                'duration_months': duration,
                'total_amount': round(monthly_amount * duration, 2),
                'payment_schedule': 'monthly',
                'per_capita_income': round(per_capita_income, 2),
                'need_level': need_level,
                'conditions': self._generate_financial_conditions(recommendation, factors)
            }

        # Economic Enablement Calculation
        if support_type in ['economic_enablement', 'both']:
            enablement_limits = limits['economic_enablement']

            # Calculate training budget based on profile and need
            base_budget = enablement_limits['training_budget']

            # Adjust budget based on education and experience level
            education_level = validated_data.get('education_level', 'High School')
            experience_years = validated_data.get('experience_years', 0)

            if education_level in ['Bachelor\'s Degree', 'Master\'s Degree', 'PhD']:
                # Higher education gets advanced training budget
                training_budget = base_budget * 1.2
                duration_months = 8
            elif education_level in ['Diploma', 'Technical Certificate']:
                # Technical background gets specialized training
                training_budget = base_budget
                duration_months = 6
            else:
                # Basic education gets foundational training
                training_budget = base_budget * 0.8
                duration_months = 9  # Longer duration for basic training

            # Adjust based on eligibility score and recommendation
            if recommendation['decision'] == 'conditional_approve':
                training_budget *= 0.75
                duration_months = min(duration_months, 6)
            elif recommendation['decision'] == 'manual_review':
                training_budget *= 0.6
                duration_months = 4

            # Determine training programs and placement probability
            recommended_programs = self._recommend_training_programs(factors, validated_data)
            job_placement_support = self._calculate_placement_support(factors, validated_data)

            details['economic_enablement'] = {
                'training_budget': round(training_budget, 2),
                'duration_months': duration_months,
                'job_placement_support': job_placement_support,
                'recommended_programs': recommended_programs,
                'placement_probability': self._estimate_placement_probability(validated_data),
                'conditions': self._generate_enablement_conditions(recommendation, factors)
            }

        return details

    def _generate_financial_conditions(self, recommendation: Dict[str, Any], factors: Dict[str, Any]) -> List[str]:
        """Generate conditions for financial support"""
        conditions = []

        if recommendation['decision'] == 'conditional_approve':
            conditions.append("Monthly income verification required")
            conditions.append("Quarterly review meetings mandatory")

        if factors.get('risk_level') == 'High':
            conditions.append("Enhanced monitoring and reporting required")

        if factors.get('validation_confidence', 1.0) < 0.8:
            conditions.append("Additional document verification required")

        return conditions

    def _generate_enablement_conditions(self, recommendation: Dict[str, Any], factors: Dict[str, Any]) -> List[str]:
        """Generate conditions for economic enablement support"""
        conditions = [
            "Minimum 80% attendance required for training programs",
            "Monthly progress reports required"
        ]

        if recommendation['decision'] == 'conditional_approve':
            conditions.append("Complete assessment before final approval")

        return conditions

    def _recommend_training_programs(self, factors: Dict[str, Any], validated_data: Dict[str, Any] = None) -> List[str]:
        """Recommend specific training programs based on profile"""
        programs = []

        if not validated_data:
            validated_data = {}

        education_level = validated_data.get('education_level', 'High School')
        experience_years = validated_data.get('experience_years', 0)
        employment_status = validated_data.get('employment_status', 'Unemployed')
        age = validated_data.get('age', 30)

        # High-demand tech programs for younger candidates
        if age <= 35 and education_level in ['Bachelor\'s Degree', 'Master\'s Degree', 'PhD']:
            programs.extend([
                "Advanced Data Analytics Certification",
                "Cloud Computing (AWS/Azure) Training",
                "Digital Marketing Professional",
                "Full-Stack Web Development"
            ])

        # Mid-level technical programs
        elif education_level in ['Diploma', 'Technical Certificate'] or (experience_years > 2 and age <= 45):
            programs.extend([
                "Digital Skills Bootcamp",
                "Project Management Professional (PMP)",
                "Customer Service Excellence",
                "Digital Administration"
            ])

        # Entry-level and foundational programs
        else:
            programs.extend([
                "Basic Computer Literacy",
                "Retail and Sales Training",
                "Food Service Certification",
                "Administrative Assistant Training"
            ])

        # Add financial literacy for everyone
        programs.append("Financial Literacy and Money Management")

        # Entrepreneurship for those with experience
        if experience_years > 3 or factors.get('financial_need_level') in ['moderate', 'low']:
            programs.append("Small Business Development Program")

        return programs[:3]  # Return top 3 recommendations

    def _calculate_placement_support(self, factors: Dict[str, Any], validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate job placement support level"""
        education_level = validated_data.get('education_level', 'High School')
        experience_years = validated_data.get('experience_years', 0)
        age = validated_data.get('age', 30)

        # Determine placement support level
        if education_level in ['Bachelor\'s Degree', 'Master\'s Degree', 'PhD'] and experience_years > 2:
            support_level = "Premium"
            services = [
                "Personal career counseling",
                "Executive job matching",
                "Interview coaching",
                "Salary negotiation support",
                "Professional networking events"
            ]
        elif education_level in ['Diploma', 'Technical Certificate'] or experience_years > 1:
            support_level = "Standard"
            services = [
                "Job matching service",
                "Resume writing assistance",
                "Interview preparation",
                "Skills assessment",
                "Job fair invitations"
            ]
        else:
            support_level = "Basic"
            services = [
                "Job search training",
                "Basic resume help",
                "Employment orientation",
                "On-the-job training placement"
            ]

        return {
            "support_level": support_level,
            "services_included": services,
            "guaranteed_interviews": 3 if support_level == "Premium" else 2 if support_level == "Standard" else 1
        }

    def _estimate_placement_probability(self, validated_data: Dict[str, Any]) -> str:
        """Estimate job placement probability"""
        education_level = validated_data.get('education_level', 'High School')
        experience_years = validated_data.get('experience_years', 0)
        age = validated_data.get('age', 30)

        score = 0.5  # Base probability

        # Education factor
        if education_level in ['Master\'s Degree', 'PhD']:
            score += 0.3
        elif education_level == 'Bachelor\'s Degree':
            score += 0.2
        elif education_level in ['Diploma', 'Technical Certificate']:
            score += 0.1

        # Experience factor
        if experience_years > 5:
            score += 0.2
        elif experience_years > 2:
            score += 0.1

        # Age factor (market preference)
        if 25 <= age <= 45:
            score += 0.1
        elif age > 50:
            score -= 0.1

        # Convert to percentage and categorize
        probability = min(0.95, max(0.15, score))

        if probability >= 0.8:
            return "High (80%+)"
        elif probability >= 0.6:
            return "Good (60-79%)"
        elif probability >= 0.4:
            return "Moderate (40-59%)"
        else:
            return "Challenging (<40%)"

    async def _generate_decision_reasoning(self, recommendation: Dict[str, Any],
                                         factors: Dict[str, Any],
                                         support_details: Dict[str, Any]) -> str:
        """Generate comprehensive reasoning for the decision"""
        try:
            prompt_template = """
            Provide a comprehensive reasoning for the social support decision based on the following analysis:

            Recommendation: {decision}
            Confidence: {confidence}

            Key Factors:
            - Eligibility Score: {eligibility_score}
            - Rule-based Eligible: {rule_based_eligible}
            - Risk Level: {risk_level}
            - Document Quality: {document_quality}
            - Financial Need Level: {financial_need_level}
            - Validation Issues: {validation_issues}

            Support Details:
            {support_details}

            Provide a clear, professional explanation that:
            1. Explains the decision rationale
            2. Highlights key factors that influenced the decision
            3. Outlines any conditions or next steps
            4. Addresses any concerns or limitations
            """

            prompt = self._create_prompt(prompt_template, {
                'decision': recommendation['decision'],
                'confidence': recommendation['confidence'],
                'eligibility_score': factors['eligibility_score'],
                'rule_based_eligible': factors['rule_based_eligible'],
                'risk_level': factors['risk_level'],
                'document_quality': factors['document_quality'],
                'financial_need_level': factors['financial_need_level'],
                'validation_issues': factors['validation_issues'],
                'support_details': str(support_details)
            })

            system_message = "You are an expert social support case worker providing clear, compassionate, and professional decision explanations."

            reasoning = await self._call_llm(prompt, system_message)
            return reasoning

        except Exception as e:
            self.logger.error(f"Decision reasoning generation failed: {str(e)}")
            return f"Decision: {recommendation['decision']}. Based on comprehensive assessment of eligibility criteria, risk factors, and validation results."

    def _create_audit_trail(self, recommendation: Dict[str, Any],
                           factors: Dict[str, Any],
                           support_details: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed audit trail for the decision"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_type': self.agent_type.value,
            'decision': recommendation['decision'],
            'confidence': recommendation['confidence'],
            'factors_considered': factors,
            'support_amounts': support_details,
            'manual_review_required': recommendation.get('requires_manual_review', False),
            'decision_criteria_version': '1.0'
        }