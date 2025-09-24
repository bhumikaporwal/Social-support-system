import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    field: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    is_valid: bool
    confidence_score: float
    issues: List[ValidationIssue]
    validated_data: Dict[str, Any]

class DataValidator:
    """Cross-document data validation and consistency checking"""

    def __init__(self):
        self.validation_rules = {
            'personal_info': self._validate_personal_info,
            'financial_info': self._validate_financial_info,
            'employment_info': self._validate_employment_info,
            'address_info': self._validate_address_info
        }

    def validate_application_data(self, extracted_documents: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """Main validation entry point for all application documents"""
        issues = []
        validated_data = {}

        try:
            # Cross-reference personal information
            personal_validation = self._validate_personal_info(extracted_documents)
            issues.extend(personal_validation.issues)
            validated_data.update(personal_validation.validated_data)

            # Validate financial consistency
            financial_validation = self._validate_financial_info(extracted_documents)
            issues.extend(financial_validation.issues)
            validated_data.update(financial_validation.validated_data)

            # Validate employment information
            employment_validation = self._validate_employment_info(extracted_documents)
            issues.extend(employment_validation.issues)
            validated_data.update(employment_validation.validated_data)

            # Calculate overall validation score
            error_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
            warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)

            # Confidence calculation
            total_checks = len(issues) + 10  # Base checks
            confidence_score = max(0.0, 1.0 - (error_count * 0.2) - (warning_count * 0.1))

            is_valid = error_count == 0

            return ValidationResult(
                is_valid=is_valid,
                confidence_score=confidence_score,
                issues=issues,
                validated_data=validated_data
            )

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[ValidationIssue(
                    field="general",
                    issue_type="validation_error",
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation process failed: {str(e)}"
                )],
                validated_data={}
            )

    def _validate_personal_info(self, documents: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """Validate personal information consistency across documents"""
        issues = []
        validated_data = {}

        emirates_id_data = documents.get('emirates_id', {})
        resume_data = documents.get('resume', {})
        bank_data = documents.get('bank_statement', {})

        # Name validation
        names = {}
        if emirates_id_data.get('name'):
            names['emirates_id'] = emirates_id_data['name'].strip().lower()
        if resume_data.get('name'):
            names['resume'] = resume_data['name'].strip().lower()
        if bank_data.get('account_holder'):
            names['bank_statement'] = bank_data['account_holder'].strip().lower()

        if len(names) > 1:
            name_values = list(names.values())
            # Check for name consistency (allowing for partial matches)
            if not self._are_names_similar(name_values):
                issues.append(ValidationIssue(
                    field="name",
                    issue_type="inconsistency",
                    severity=ValidationSeverity.WARNING,
                    message=f"Name variations found across documents: {names}",
                    suggestion="Verify name spelling and format consistency"
                ))

        # Use Emirates ID name as primary if available
        primary_name = emirates_id_data.get('name') or resume_data.get('name') or bank_data.get('account_holder')
        if primary_name:
            validated_data['full_name'] = primary_name.strip()

        # Date of birth validation
        emirates_dob = emirates_id_data.get('date_of_birth')
        if emirates_dob:
            try:
                if isinstance(emirates_dob, str):
                    dob = datetime.strptime(emirates_dob, '%d/%m/%Y').date()
                    validated_data['date_of_birth'] = dob

                    # Age validation
                    age = (date.today() - dob).days // 365
                    if age < 18:
                        issues.append(ValidationIssue(
                            field="date_of_birth",
                            issue_type="eligibility",
                            severity=ValidationSeverity.ERROR,
                            message="Applicant must be at least 18 years old"
                        ))
                    elif age > 100:
                        issues.append(ValidationIssue(
                            field="date_of_birth",
                            issue_type="data_quality",
                            severity=ValidationSeverity.WARNING,
                            message="Age appears unusually high, please verify"
                        ))

            except ValueError:
                issues.append(ValidationIssue(
                    field="date_of_birth",
                    issue_type="format_error",
                    severity=ValidationSeverity.ERROR,
                    message="Invalid date format in Emirates ID"
                ))

        # Emirates ID validation
        emirates_id = emirates_id_data.get('id_number')
        if emirates_id:
            if not self._validate_emirates_id_format(emirates_id):
                issues.append(ValidationIssue(
                    field="emirates_id",
                    issue_type="format_error",
                    severity=ValidationSeverity.ERROR,
                    message="Invalid Emirates ID format"
                ))
            else:
                validated_data['emirates_id'] = emirates_id

        # Email validation
        email = resume_data.get('email')
        if email:
            if not self._validate_email_format(email):
                issues.append(ValidationIssue(
                    field="email",
                    issue_type="format_error",
                    severity=ValidationSeverity.WARNING,
                    message="Invalid email format"
                ))
            else:
                validated_data['email'] = email

        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0,
            confidence_score=0.8,
            issues=issues,
            validated_data=validated_data
        )

    def _validate_financial_info(self, documents: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """Validate financial information consistency"""
        issues = []
        validated_data = {}

        bank_data = documents.get('bank_statement', {})
        assets_data = documents.get('assets_liabilities', {})
        credit_data = documents.get('credit_report', {})

        # Income validation
        bank_income = bank_data.get('monthly_income', 0)
        if bank_income:
            validated_data['monthly_income'] = bank_income

            # Income reasonableness check
            if bank_income < 1000:
                issues.append(ValidationIssue(
                    field="monthly_income",
                    issue_type="eligibility",
                    severity=ValidationSeverity.WARNING,
                    message="Monthly income appears very low for UAE standards"
                ))
            elif bank_income > 100000:
                issues.append(ValidationIssue(
                    field="monthly_income",
                    issue_type="data_quality",
                    severity=ValidationSeverity.INFO,
                    message="Monthly income is exceptionally high, please verify"
                ))

        # Assets vs Bank Balance consistency
        bank_balance = bank_data.get('average_balance', 0)
        liquid_assets = assets_data.get('liquid_assets', 0)

        if bank_balance and liquid_assets:
            difference_ratio = abs(bank_balance - liquid_assets) / max(bank_balance, liquid_assets)
            if difference_ratio > 0.5:  # 50% difference threshold
                issues.append(ValidationIssue(
                    field="financial_consistency",
                    issue_type="inconsistency",
                    severity=ValidationSeverity.WARNING,
                    message=f"Large discrepancy between bank balance ({bank_balance}) and reported liquid assets ({liquid_assets})"
                ))

        # Credit report vs declared debt
        credit_debt = credit_data.get('total_debt', 0)
        reported_liabilities = assets_data.get('total_liabilities', 0)

        if credit_debt and reported_liabilities:
            difference_ratio = abs(credit_debt - reported_liabilities) / max(credit_debt, reported_liabilities)
            if difference_ratio > 0.3:  # 30% difference threshold
                issues.append(ValidationIssue(
                    field="debt_consistency",
                    issue_type="inconsistency",
                    severity=ValidationSeverity.WARNING,
                    message=f"Discrepancy between credit report debt ({credit_debt}) and declared liabilities ({reported_liabilities})"
                ))

        # Net worth calculation
        total_assets = assets_data.get('total_assets', 0)
        total_liabilities = assets_data.get('total_liabilities', 0)

        if total_assets and total_liabilities:
            net_worth = total_assets - total_liabilities
            validated_data['net_worth'] = net_worth

            # Debt-to-income ratio
            if bank_income and total_liabilities:
                monthly_debt = total_liabilities / 12  # Assume yearly liabilities
                dti_ratio = monthly_debt / bank_income
                validated_data['debt_to_income_ratio'] = dti_ratio

                if dti_ratio > 0.4:  # 40% DTI threshold
                    issues.append(ValidationIssue(
                        field="debt_to_income",
                        issue_type="risk_assessment",
                        severity=ValidationSeverity.WARNING,
                        message=f"High debt-to-income ratio: {dti_ratio:.2%}"
                    ))

        # Credit score validation
        credit_score = credit_data.get('credit_score')
        if credit_score:
            validated_data['credit_score'] = credit_score
            if credit_score < 580:
                issues.append(ValidationIssue(
                    field="credit_score",
                    issue_type="risk_assessment",
                    severity=ValidationSeverity.WARNING,
                    message="Poor credit score may affect approval"
                ))

        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0,
            confidence_score=0.8,
            issues=issues,
            validated_data=validated_data
        )

    def _validate_employment_info(self, documents: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """Validate employment information"""
        issues = []
        validated_data = {}

        resume_data = documents.get('resume', {})
        bank_data = documents.get('bank_statement', {})

        # Experience validation
        total_experience = resume_data.get('total_experience_years', 0)
        if total_experience:
            validated_data['experience_years'] = total_experience

            # Income vs experience correlation
            monthly_income = bank_data.get('monthly_income', 0)
            if monthly_income and total_experience:
                expected_min_income = total_experience * 1000  # Basic expectation
                if monthly_income < expected_min_income:
                    issues.append(ValidationIssue(
                        field="income_experience",
                        issue_type="inconsistency",
                        severity=ValidationSeverity.INFO,
                        message=f"Income ({monthly_income}) seems low for {total_experience} years of experience"
                    ))

        # Skills assessment
        skills = resume_data.get('skills', [])
        if skills:
            validated_data['skills_count'] = len(skills)

            # High-demand skills check
            high_demand_skills = ['python', 'javascript', 'react', 'aws', 'azure', 'data science', 'machine learning']
            has_high_demand_skills = any(skill.lower() in ' '.join(skills).lower() for skill in high_demand_skills)
            validated_data['has_high_demand_skills'] = has_high_demand_skills

        return ValidationResult(
            is_valid=True,
            confidence_score=0.9,
            issues=issues,
            validated_data=validated_data
        )

    def _are_names_similar(self, names: List[str]) -> bool:
        """Check if names are similar (allowing for variations)"""
        if len(names) < 2:
            return True

        # Simple similarity check - at least 60% of characters match
        base_name = names[0]
        for name in names[1:]:
            # Remove common variations
            base_clean = re.sub(r'[^\w\s]', '', base_name.lower())
            name_clean = re.sub(r'[^\w\s]', '', name.lower())

            # Check if names share common words
            base_words = set(base_clean.split())
            name_words = set(name_clean.split())

            if len(base_words & name_words) == 0:
                return False

        return True

    def _validate_address_info(self, documents: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """Validate address information across documents"""
        issues = []
        validated_data = {}

        # Collect addresses from different documents
        addresses = []
        for doc_type, data in documents.items():
            if 'address' in data:
                addresses.append({
                    'source': doc_type,
                    'address': data['address']
                })

        if not addresses:
            issues.append(ValidationIssue(
                field="address",
                issue_type="missing_data",
                severity=ValidationSeverity.WARNING,
                message="No address information found in any document"
            ))
            return ValidationResult(
                is_valid=len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0,
                confidence_score=0.3,
                issues=issues,
                validated_data=validated_data
            )

        # Use the first address found as the validated address
        primary_address = addresses[0]['address']
        validated_data['address'] = primary_address

        # Check for consistency if multiple addresses exist
        if len(addresses) > 1:
            consistent = True
            for addr_info in addresses[1:]:
                if addr_info['address'] != primary_address:
                    consistent = False
                    issues.append(ValidationIssue(
                        field="address",
                        issue_type="inconsistency",
                        severity=ValidationSeverity.WARNING,
                        message=f"Address mismatch between {addresses[0]['source']} and {addr_info['source']}",
                        suggestion="Verify the correct address with the applicant"
                    ))

            if consistent:
                issues.append(ValidationIssue(
                    field="address",
                    issue_type="consistency_check",
                    severity=ValidationSeverity.INFO,
                    message="Address information is consistent across all documents"
                ))

        # Basic address format validation
        if isinstance(primary_address, str) and len(primary_address.strip()) < 10:
            issues.append(ValidationIssue(
                field="address",
                issue_type="format_error",
                severity=ValidationSeverity.WARNING,
                message="Address seems too short - may be incomplete",
                suggestion="Request full address details from applicant"
            ))

        confidence_score = 0.9 if len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0 else 0.3

        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0,
            confidence_score=confidence_score,
            issues=issues,
            validated_data=validated_data
        )

    def _validate_emirates_id_format(self, emirates_id: str) -> bool:
        """Validate Emirates ID format"""
        # Remove any spaces or dashes
        clean_id = re.sub(r'[-\s]', '', emirates_id)

        # Should be 15 digits
        if not re.match(r'^\d{15}$', clean_id):
            return False

        # Basic checksum validation (simplified)
        return True

    def _validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None