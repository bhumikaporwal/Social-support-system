from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentType, AgentResult

class EconomicEnablementAgent(BaseAgent):
    """Agent responsible for economic enablement recommendations"""

    def __init__(self, llm_client=None, langfuse_client=None):
        super().__init__(AgentType.ECONOMIC_ENABLEMENT, llm_client, langfuse_client)
        self.training_programs = self._load_training_programs()
        self.job_market_data = self._load_job_market_data()
        self.skills_mapping = self._load_skills_mapping()

    def _load_training_programs(self) -> Dict[str, Dict[str, Any]]:
        """Load available training programs"""
        return {
            "digital_skills": {
                "name": "Digital Skills Bootcamp",
                "duration": "3 months",
                "cost": 5000,
                "skills": ["computer_literacy", "microsoft_office", "digital_marketing"],
                "job_prospects": ["administrative_assistant", "marketing_coordinator", "data_entry_specialist"],
                "success_rate": 0.85
            },
            "programming": {
                "name": "Software Development Program",
                "duration": "6 months",
                "cost": 12000,
                "skills": ["python", "javascript", "web_development", "database_management"],
                "job_prospects": ["junior_developer", "web_developer", "software_engineer"],
                "success_rate": 0.75
            },
            "data_analysis": {
                "name": "Data Analytics Certification",
                "duration": "4 months",
                "cost": 8000,
                "skills": ["excel_advanced", "sql", "python", "data_visualization"],
                "job_prospects": ["data_analyst", "business_analyst", "reporting_specialist"],
                "success_rate": 0.80
            },
            "entrepreneurship": {
                "name": "Business Development Program",
                "duration": "2 months",
                "cost": 3000,
                "skills": ["business_planning", "financial_management", "marketing", "leadership"],
                "job_prospects": ["business_owner", "consultant", "project_manager"],
                "success_rate": 0.60
            },
            "healthcare": {
                "name": "Healthcare Support Certification",
                "duration": "5 months",
                "cost": 7000,
                "skills": ["patient_care", "medical_terminology", "healthcare_administration"],
                "job_prospects": ["healthcare_assistant", "medical_receptionist", "patient_coordinator"],
                "success_rate": 0.90
            },
            "logistics": {
                "name": "Supply Chain and Logistics",
                "duration": "3 months",
                "cost": 4500,
                "skills": ["inventory_management", "logistics_coordination", "supply_chain_basics"],
                "job_prospects": ["logistics_coordinator", "warehouse_manager", "supply_chain_analyst"],
                "success_rate": 0.75
            }
        }

    def _load_job_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Load current job market data for UAE"""
        return {
            "high_demand_sectors": [
                "technology",
                "healthcare",
                "logistics",
                "renewable_energy",
                "tourism",
                "financial_services"
            ],
            "salary_ranges": {
                "junior_developer": {"min": 8000, "max": 15000},
                "data_analyst": {"min": 10000, "max": 18000},
                "healthcare_assistant": {"min": 6000, "max": 12000},
                "logistics_coordinator": {"min": 7000, "max": 14000},
                "administrative_assistant": {"min": 5000, "max": 10000},
                "marketing_coordinator": {"min": 6000, "max": 12000}
            },
            "growth_projections": {
                "technology": 0.25,  # 25% growth expected
                "healthcare": 0.20,
                "logistics": 0.18,
                "renewable_energy": 0.30
            }
        }

    def _load_skills_mapping(self) -> Dict[str, List[str]]:
        """Load mapping of current skills to potential career paths"""
        return {
            "communication": ["sales", "customer_service", "marketing", "training"],
            "analytical_thinking": ["data_analysis", "research", "consulting", "finance"],
            "technical_aptitude": ["it_support", "software_development", "engineering"],
            "leadership": ["management", "project_management", "team_leadership"],
            "creativity": ["design", "marketing", "content_creation", "innovation"],
            "problem_solving": ["consulting", "engineering", "research", "management"]
        }

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Generate economic enablement recommendations"""
        try:
            validated_data = input_data.get('validated_data', {})
            eligibility_data = input_data.get('eligibility_assessment', {})
            final_recommendation = input_data.get('final_recommendation', {})

            # Extract relevant profile information
            profile = self._extract_profile(validated_data)

            # Analyze current skills and experience
            skills_analysis = self._analyze_skills_and_experience(profile)

            # Generate training recommendations
            training_recommendations = self._generate_training_recommendations(profile, skills_analysis)

            # Generate job matching recommendations
            job_recommendations = self._generate_job_recommendations(profile, skills_analysis)

            # Create career development pathway
            career_pathway = self._create_career_pathway(profile, training_recommendations, job_recommendations)

            # Generate financial projections
            financial_projections = self._calculate_financial_projections(career_pathway, profile)

            # Generate LLM-based personalized advice
            personalized_advice = await self._generate_personalized_advice(
                profile, training_recommendations, job_recommendations
            )

            # Create implementation timeline
            implementation_plan = self._create_implementation_plan(training_recommendations, career_pathway)

            return AgentResult(
                success=True,
                data={
                    "profile_analysis": profile,
                    "skills_analysis": skills_analysis,
                    "training_recommendations": training_recommendations,
                    "job_recommendations": job_recommendations,
                    "career_pathway": career_pathway,
                    "financial_projections": financial_projections,
                    "implementation_plan": implementation_plan
                },
                reasoning=personalized_advice,
                confidence=0.8
            )

        except Exception as e:
            self.logger.error(f"Economic enablement recommendation failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                reasoning=f"Economic enablement recommendation failed: {str(e)}",
                confidence=0.0,
                errors=[str(e)]
            )

    def _extract_profile(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant profile information for career recommendations"""
        return {
            'age': self._calculate_age(validated_data.get('date_of_birth')),
            'education_level': validated_data.get('education_level'),
            'experience_years': validated_data.get('experience_years', 0),
            'current_income': validated_data.get('monthly_income', 0),
            'skills': validated_data.get('skills', []),
            'has_high_demand_skills': validated_data.get('has_high_demand_skills', False),
            'employment_status': validated_data.get('employment_status'),
            'family_size': validated_data.get('family_size', 1),
            'financial_obligations': validated_data.get('total_liabilities', 0)
        }

    def _calculate_age(self, date_of_birth) -> int:
        """Calculate age from date of birth"""
        if not date_of_birth:
            return 30  # Default assumption

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
            return 30

    def _analyze_skills_and_experience(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current skills and experience"""
        current_skills = profile.get('skills', [])
        experience_years = profile.get('experience_years', 0)

        # Categorize skills
        skill_categories = {
            'technical': [],
            'soft': [],
            'domain_specific': []
        }

        technical_keywords = ['python', 'javascript', 'sql', 'excel', 'programming', 'software', 'computer']
        soft_keywords = ['leadership', 'communication', 'teamwork', 'problem', 'management']

        for skill in current_skills:
            skill_lower = skill.lower()
            if any(keyword in skill_lower for keyword in technical_keywords):
                skill_categories['technical'].append(skill)
            elif any(keyword in skill_lower for keyword in soft_keywords):
                skill_categories['soft'].append(skill)
            else:
                skill_categories['domain_specific'].append(skill)

        # Assess skill level
        if experience_years < 2:
            skill_level = 'beginner'
        elif experience_years < 5:
            skill_level = 'intermediate'
        elif experience_years < 10:
            skill_level = 'experienced'
        else:
            skill_level = 'expert'

        # Identify skill gaps
        skill_gaps = self._identify_skill_gaps(skill_categories, profile)

        return {
            'skill_categories': skill_categories,
            'skill_level': skill_level,
            'experience_years': experience_years,
            'skill_gaps': skill_gaps,
            'transferable_skills': skill_categories['soft'],
            'development_potential': self._assess_development_potential(profile)
        }

    def _identify_skill_gaps(self, skill_categories: Dict[str, List[str]], profile: Dict[str, Any]) -> List[str]:
        """Identify skill gaps for career advancement"""
        gaps = []

        # Technical skill gaps
        if len(skill_categories['technical']) < 3:
            gaps.append("Limited technical skills - consider digital literacy training")

        # High-demand skills gaps
        high_demand_skills = ['data_analysis', 'digital_marketing', 'project_management', 'programming']
        current_skills_lower = [skill.lower() for skill in sum(skill_categories.values(), [])]

        for skill in high_demand_skills:
            if not any(skill in current_skill for current_skill in current_skills_lower):
                gaps.append(f"Missing high-demand skill: {skill}")

        # Industry-specific gaps
        if profile['age'] < 35 and profile['experience_years'] < 5:
            gaps.append("Limited professional experience - consider internship or mentorship programs")

        return gaps

    def _assess_development_potential(self, profile: Dict[str, Any]) -> str:
        """Assess individual's potential for career development"""
        age = profile['age']
        education = profile.get('education_level', '')
        experience = profile['experience_years']

        if age < 30 and experience < 3:
            return 'high'  # Young with room to grow
        elif age < 40 and ('bachelor' in education.lower() or 'degree' in education.lower()):
            return 'high'  # Good education foundation
        elif age < 50 and experience > 5:
            return 'medium'  # Experienced but may need reskilling
        else:
            return 'medium'  # Focus on specialized skills

    def _generate_training_recommendations(self, profile: Dict[str, Any], skills_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized training recommendations"""
        recommendations = []
        training_programs = self.training_programs

        age = profile['age']
        income = profile['current_income']
        experience = profile['experience_years']
        has_tech_skills = len(skills_analysis['skill_categories']['technical']) > 0

        # Priority 1: Address immediate skill gaps
        if not has_tech_skills and age < 45:
            recommendations.append({
                **training_programs['digital_skills'],
                'priority': 'high',
                'reason': 'Essential digital skills for modern workplace',
                'roi_months': 6
            })

        # Priority 2: High-growth potential programs
        if income < 8000 and experience < 10:
            if has_tech_skills:
                recommendations.append({
                    **training_programs['programming'],
                    'priority': 'high',
                    'reason': 'High salary potential in growing tech sector',
                    'roi_months': 12
                })
            else:
                recommendations.append({
                    **training_programs['data_analysis'],
                    'priority': 'medium',
                    'reason': 'Growing demand for data skills across industries',
                    'roi_months': 8
                })

        # Priority 3: Stable employment sectors
        recommendations.append({
            **training_programs['healthcare'],
            'priority': 'medium',
            'reason': 'Stable employment with growing demand',
            'roi_months': 10
        })

        # Priority 4: Entrepreneurship for experienced individuals
        if experience > 5 or age > 35:
            recommendations.append({
                **training_programs['entrepreneurship'],
                'priority': 'low',
                'reason': 'Leverage existing experience for business opportunities',
                'roi_months': 18
            })

        # Sort by priority and potential ROI
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: (priority_order[x['priority']], -x['roi_months']))

        return recommendations[:3]  # Return top 3 recommendations

    def _generate_job_recommendations(self, profile: Dict[str, Any], skills_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate job matching recommendations"""
        recommendations = []
        job_market = self.job_market_data
        salary_ranges = job_market['salary_ranges']

        current_income = profile['current_income']
        skill_level = skills_analysis['skill_level']
        technical_skills = skills_analysis['skill_categories']['technical']

        # Match based on current skills
        if technical_skills:
            if skill_level in ['intermediate', 'experienced']:
                recommendations.append({
                    'job_title': 'Data Analyst',
                    'salary_range': salary_ranges['data_analyst'],
                    'match_score': 0.8,
                    'required_skills': ['excel', 'data_analysis', 'reporting'],
                    'growth_potential': 'high'
                })

        # Entry-level opportunities
        recommendations.append({
            'job_title': 'Administrative Assistant',
            'salary_range': salary_ranges['administrative_assistant'],
            'match_score': 0.7,
            'required_skills': ['computer_literacy', 'communication', 'organization'],
            'growth_potential': 'medium'
        })

        # Healthcare opportunities (stable sector)
        recommendations.append({
            'job_title': 'Healthcare Assistant',
            'salary_range': salary_ranges['healthcare_assistant'],
            'match_score': 0.6,
            'required_skills': ['patient_care', 'communication', 'attention_to_detail'],
            'growth_potential': 'high'
        })

        # Filter recommendations that improve income
        filtered_recommendations = [
            rec for rec in recommendations
            if rec['salary_range']['min'] > current_income * 0.8  # At least maintain current income
        ]

        return filtered_recommendations[:3]

    def _create_career_pathway(self, profile: Dict[str, Any], training_recs: List[Dict[str, Any]], job_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a structured career development pathway"""
        pathway = {
            'immediate_actions': [],
            'short_term_goals': [],  # 3-6 months
            'medium_term_goals': [],  # 6-18 months
            'long_term_vision': []   # 18+ months
        }

        # Immediate actions
        if training_recs:
            pathway['immediate_actions'].append(f"Enroll in {training_recs[0]['name']}")
            pathway['immediate_actions'].append("Update CV with current skills and experience")
            pathway['immediate_actions'].append("Set up LinkedIn profile")

        # Short-term goals
        if training_recs:
            pathway['short_term_goals'].append(f"Complete {training_recs[0]['name']} certification")
            pathway['short_term_goals'].append("Build portfolio of projects/work samples")

        # Medium-term goals
        if job_recs:
            target_job = job_recs[0]
            pathway['medium_term_goals'].append(f"Secure position as {target_job['job_title']}")
            pathway['medium_term_goals'].append(f"Achieve salary of {target_job['salary_range']['min']} AED")

        # Long-term vision
        pathway['long_term_vision'].append("Achieve financial stability and career growth")
        if len(training_recs) > 1:
            pathway['long_term_vision'].append(f"Consider advanced training in {training_recs[1]['name']}")

        return pathway

    def _calculate_financial_projections(self, career_pathway: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial impact projections"""
        current_income = profile['current_income']

        projections = {
            'current_monthly_income': current_income,
            'projected_6_month_income': current_income * 1.1,  # Modest increase
            'projected_12_month_income': current_income * 1.4,  # After training completion
            'projected_24_month_income': current_income * 1.8,  # With job transition
            'total_investment_required': 0,
            'roi_timeline': '12-18 months'
        }

        return projections

    def _create_implementation_plan(self, training_recs: List[Dict[str, Any]], career_pathway: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed implementation timeline"""
        plan = {
            'phase_1': {
                'duration': '0-3 months',
                'actions': career_pathway.get('immediate_actions', []),
                'milestones': ['Training enrollment', 'CV update', 'Skill assessment']
            },
            'phase_2': {
                'duration': '3-6 months',
                'actions': career_pathway.get('short_term_goals', []),
                'milestones': ['Training completion', 'Portfolio development', 'Network building']
            },
            'phase_3': {
                'duration': '6-18 months',
                'actions': career_pathway.get('medium_term_goals', []),
                'milestones': ['Job applications', 'Interview preparation', 'Position securing']
            },
            'ongoing_support': {
                'mentorship': 'Monthly check-ins with career counselor',
                'job_placement': 'Access to job placement services',
                'continued_learning': 'Subscription to online learning platforms'
            }
        }

        return plan

    async def _generate_personalized_advice(self, profile: Dict[str, Any], training_recs: List[Dict[str, Any]], job_recs: List[Dict[str, Any]]) -> str:
        """Generate personalized career advice using LLM"""
        try:
            prompt_template = """
            Provide personalized career advice for the following individual:

            Profile:
            - Age: {age}
            - Current Income: {current_income} AED/month
            - Experience: {experience_years} years
            - Education: {education_level}
            - Family Size: {family_size}

            Recommended Training:
            {training_recommendations}

            Job Opportunities:
            {job_opportunities}

            Provide motivational, practical advice that:
            1. Acknowledges their current situation
            2. Highlights the potential for improvement
            3. Provides specific next steps
            4. Addresses potential challenges
            5. Emphasizes the support available
            """

            training_summary = "\n".join([f"- {rec['name']}: {rec['reason']}" for rec in training_recs[:2]])
            job_summary = "\n".join([f"- {rec['job_title']}: {rec['salary_range']['min']}-{rec['salary_range']['max']} AED" for rec in job_recs[:2]])

            prompt = self._create_prompt(prompt_template, {
                'age': profile['age'],
                'current_income': profile['current_income'],
                'experience_years': profile['experience_years'],
                'education_level': profile.get('education_level', 'Not specified'),
                'family_size': profile['family_size'],
                'training_recommendations': training_summary,
                'job_opportunities': job_summary
            })

            system_message = "You are a compassionate career counselor providing personalized advice for economic enablement. Be encouraging and practical."

            advice = await self._call_llm(prompt, system_message)
            return advice

        except Exception as e:
            self.logger.error(f"Personalized advice generation failed: {str(e)}")
            return "Based on your profile and experience, we've identified several opportunities for career advancement through targeted training and job placement support."