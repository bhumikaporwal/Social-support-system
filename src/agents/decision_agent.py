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
            'financial_need_level': self._assess_financial_need(validation_data.get('validated_data', {})),
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

        # Cannot approve if critical validation errors exist
        if critical_errors > 0:
            return {
                'decision': 'decline',
                'reason': 'Critical validation errors found',
                'confidence': 0.9,
                'requires_manual_review': False
            }

        # Cannot approve without basic eligibility
        if not rule_based_eligible:
            return {
                'decision': 'decline',
                'reason': 'Does not meet basic eligibility criteria',
                'confidence': 0.8,
                'requires_manual_review': False
            }

        # Document quality check
        if document_quality < 0.5:
            return {
                'decision': 'manual_review',
                'reason': 'Poor document quality requires manual verification',
                'confidence': 0.7,
                'requires_manual_review': True
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
            return {}

        details = {}
        limits = self.approval_limits

        need_level = factors.get('financial_need_level', 'moderate')
        eligibility_score = factors.get('eligibility_score', 0.5)

        # Financial Support Calculation
        if support_type in ['financial', 'both']:
            financial_limits = limits['financial_support']

            # Base amount based on need level
            need_multipliers = {
                'critical': 1.0,
                'high': 0.8,
                'moderate': 0.6,
                'low': 0.4
            }

            base_amount = financial_limits['monthly_amount_range'][1] * need_multipliers.get(need_level, 0.6)

            # Adjust based on eligibility score
            score_multiplier = min(eligibility_score * 1.2, 1.0)
            monthly_amount = base_amount * score_multiplier

            # Determine duration
            duration = financial_limits['duration_months_range'][1] if need_level == 'critical' else financial_limits['duration_months_range'][0]

            if recommendation['decision'] == 'conditional_approve':
                monthly_amount *= 0.7  # Reduce amount for conditional approval
                duration = min(duration, 6)  # Limit duration

            details['financial_support'] = {
                'monthly_amount': round(monthly_amount, 2),
                'duration_months': duration,
                'total_amount': round(monthly_amount * duration, 2),
                'payment_schedule': 'monthly',
                'conditions': self._generate_financial_conditions(recommendation, factors)
            }

        # Economic Enablement Calculation
        if support_type in ['economic_enablement', 'both']:
            enablement_limits = limits['economic_enablement']

            training_budget = enablement_limits['training_budget']
            if recommendation['decision'] == 'conditional_approve':
                training_budget *= 0.8

            details['economic_enablement'] = {
                'training_budget': training_budget,
                'duration_months': enablement_limits['duration_months'],
                'job_placement_support': enablement_limits['job_placement_support'],
                'recommended_programs': self._recommend_training_programs(factors),
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

    def _recommend_training_programs(self, factors: Dict[str, Any]) -> List[str]:
        """Recommend specific training programs based on profile"""
        programs = []

        # This would integrate with a job market analysis system
        # For demo, provide basic recommendations

        if factors.get('financial_need_level') in ['critical', 'high']:
            programs.append("Fast-track vocational certification")
            programs.append("Digital skills bootcamp")

        programs.append("Entrepreneurship development program")
        programs.append("Financial literacy course")

        return programs

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