import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from faker import Faker
from typing import List, Dict, Any
import uuid
import os
from pathlib import Path
import re

# Initialize Faker with Arabic locale for UAE context
fake = Faker(['ar_AE', 'en_US'])
fake_ar = Faker('ar_AE')
fake_en = Faker('en_US')

# UAE-specific data
UAE_EMIRATES = ['Abu Dhabi', 'Dubai', 'Sharjah', 'Ajman', 'Umm Al Quwain', 'Ras Al Khaimah', 'Fujairah']
UAE_NATIONALITIES = ['UAE', 'Egyptian', 'Indian', 'Pakistani', 'Bangladeshi', 'Filipino', 'Syrian', 'Lebanese', 'Jordanian', 'Palestinian', 'British', 'American']
BANKS = ['Emirates NBD', 'ADCB', 'FAB', 'Mashreq Bank', 'CBD', 'HSBC UAE', 'Citibank UAE', 'NBF', 'RAKBANK']
EDUCATION_LEVELS = ['High School', 'Diploma', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD', 'Vocational Training', 'Technical Certificate']
EMPLOYMENT_STATUS = ['Employed', 'Unemployed', 'Self-employed', 'Part-time', 'Contract Worker', 'Retired', 'Student', 'Freelancer']
MARITAL_STATUS = ['Single', 'Married', 'Divorced', 'Widowed']

# Enhanced job categories for better skill matching
JOB_CATEGORIES = {
    'technology': ['Software Developer', 'Data Analyst', 'IT Support', 'Web Developer', 'System Administrator', 'Cybersecurity Specialist'],
    'healthcare': ['Nurse', 'Medical Assistant', 'Pharmacist', 'Healthcare Administrator', 'Laboratory Technician'],
    'education': ['Teacher', 'Training Coordinator', 'Academic Advisor', 'Education Consultant'],
    'business': ['Sales Manager', 'Marketing Specialist', 'Business Analyst', 'Project Manager', 'Customer Service Representative'],
    'hospitality': ['Hotel Manager', 'Restaurant Server', 'Event Coordinator', 'Tourism Guide'],
    'construction': ['Construction Worker', 'Electrician', 'Plumber', 'Project Supervisor'],
    'retail': ['Shop Assistant', 'Store Manager', 'Cashier', 'Inventory Specialist'],
    'finance': ['Accountant', 'Financial Advisor', 'Bank Teller', 'Insurance Agent']
}

# Skills mapped to job categories
SKILL_CATEGORIES = {
    'technical': {
        'technology': ['Python', 'JavaScript', 'Java', 'SQL', 'React', 'Node.js', 'AWS', 'Docker', 'Git'],
        'finance': ['Excel', 'QuickBooks', 'SAP', 'Financial Analysis', 'Accounting Software'],
        'general': ['Microsoft Office', 'PowerPoint', 'Data Entry', 'Computer Literacy']
    },
    'soft': ['Communication', 'Leadership', 'Teamwork', 'Problem Solving', 'Time Management', 'Project Management', 'Customer Service', 'Adaptability'],
    'languages': ['Arabic', 'English', 'Hindi', 'Urdu', 'French', 'German', 'Tagalog', 'Bengali']
}

class SyntheticDataGenerator:
    """Generate synthetic test data for the social support system"""

    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_emirates_id(self, birth_year: int = None) -> str:
        """Generate a realistic Emirates ID format with proper validation"""
        # Format: 784-YYYY-NNNNNNN-N
        if birth_year is None:
            birth_year = random.randint(1960, 2005)

        # Generate sequence number (more realistic patterns)
        sequence = random.randint(1000000, 9999999)

        # Generate check digit using simple algorithm
        id_without_check = f"784{birth_year}{sequence}"
        check_digit = self._calculate_check_digit(id_without_check)

        return f"784-{birth_year}-{sequence}-{check_digit}"

    def _calculate_check_digit(self, id_number: str) -> int:
        """Calculate check digit for Emirates ID (simplified algorithm)"""
        total = sum(int(digit) * (i + 1) for i, digit in enumerate(id_number))
        return total % 10

    def generate_applicant_data(self, profile_type: str = 'eligible') -> Dict[str, Any]:
        """Generate synthetic applicant data with realistic patterns"""
        gender = random.choice(['Male', 'Female'])
        nationality = random.choice(UAE_NATIONALITIES)

        # Age distribution based on profile type
        age_ranges = {
            'eligible': (22, 45),         # Younger adults with potential
            'borderline': (30, 55),       # Mid-career professionals
            'high_income': (35, 60),      # Experienced professionals
            'unemployed': (25, 50),       # Mixed ages for unemployed
            'elderly': (60, 70),          # Senior citizens
            'young_graduate': (22, 28)    # Recent graduates
        }

        min_age, max_age = age_ranges.get(profile_type, (18, 70))
        birth_date = fake.date_of_birth(minimum_age=min_age, maximum_age=max_age)
        birth_year = birth_date.year

        # Generate names based on nationality
        if nationality in ['UAE', 'Syrian', 'Lebanese', 'Jordanian', 'Palestinian']:
            if gender == 'Male':
                first_name = random.choice(['Ahmed', 'Mohammed', 'Ali', 'Omar', 'Khalid', 'Saeed', 'Rashid', 'Hamad'])
                last_name = random.choice(['Al Maktoum', 'Al Nahyan', 'Al Qasimi', 'Al Nuaimi', 'Al Mazrouei', 'Al Shamsi'])
            else:
                first_name = random.choice(['Fatima', 'Aisha', 'Mariam', 'Sara', 'Noura', 'Amna', 'Shaikha', 'Moza'])
                last_name = random.choice(['Al Maktoum', 'Al Nahyan', 'Al Qasimi', 'Al Nuaimi', 'Al Mazrouei', 'Al Shamsi'])
        elif nationality in ['Indian', 'Pakistani', 'Bangladeshi']:
            first_name = fake_en.first_name_male() if gender == 'Male' else fake_en.first_name_female()
            last_name = random.choice(['Khan', 'Sharma', 'Singh', 'Kumar', 'Ali', 'Ahmed', 'Rahman', 'Patel'])
        else:
            first_name = fake_en.first_name_male() if gender == 'Male' else fake_en.first_name_female()
            last_name = fake_en.last_name()

        # Education level based on profile and age
        education_weights = self._get_education_weights(profile_type, birth_date)
        education_level = random.choices(
            list(education_weights.keys()),
            weights=list(education_weights.values())
        )[0]

        # Marital status based on age and gender
        marital_weights = self._get_marital_weights(birth_date, gender)
        marital_status = random.choices(
            list(marital_weights.keys()),
            weights=list(marital_weights.values())
        )[0]

        # Family size based on marital status and age
        family_size = self._calculate_family_size(marital_status, birth_date)
        dependents = max(0, family_size - (2 if marital_status == 'Married' else 1))

        return {
            'emirates_id': self.generate_emirates_id(birth_year),
            'first_name': first_name,
            'last_name': last_name,
            'full_name': f"{first_name} {last_name}",
            'date_of_birth': birth_date.isoformat(),
            'age': (datetime.now().date() - birth_date).days // 365,
            'nationality': nationality,
            'gender': gender,
            'email': f"{first_name.lower()}.{last_name.lower().replace(' ', '')}@{random.choice(['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com'])}",
            'phone': f"+971{random.choice([50, 52, 54, 55, 56, 58])}{random.randint(1000000, 9999999)}",
            'address': self._generate_uae_address(),
            'emirate': random.choice(UAE_EMIRATES),
            'education_level': education_level,
            'marital_status': marital_status,
            'family_size': family_size,
            'dependents': dependents
        }

    def _get_education_weights(self, profile_type: str, birth_date: date) -> Dict[str, float]:
        """Get education level weights based on profile type and age"""
        age = (datetime.now().date() - birth_date).days // 365

        if profile_type == 'high_income':
            return {'Bachelor\'s Degree': 0.4, 'Master\'s Degree': 0.3, 'PhD': 0.2, 'Diploma': 0.1}
        elif profile_type == 'borderline':
            return {'High School': 0.2, 'Diploma': 0.3, 'Bachelor\'s Degree': 0.4, 'Master\'s Degree': 0.1}
        else:  # eligible, unemployed
            if age < 30:
                return {'High School': 0.3, 'Diploma': 0.3, 'Bachelor\'s Degree': 0.3, 'Vocational Training': 0.1}
            else:
                return {'High School': 0.4, 'Diploma': 0.2, 'Bachelor\'s Degree': 0.2, 'Vocational Training': 0.2}

    def _get_marital_weights(self, birth_date: date, gender: str) -> Dict[str, float]:
        """Get marital status weights based on age and gender"""
        age = (datetime.now().date() - birth_date).days // 365

        if age < 25:
            return {'Single': 0.8, 'Married': 0.2}
        elif age < 35:
            return {'Single': 0.4, 'Married': 0.6}
        elif age < 50:
            return {'Single': 0.2, 'Married': 0.7, 'Divorced': 0.1}
        else:
            return {'Single': 0.1, 'Married': 0.7, 'Divorced': 0.1, 'Widowed': 0.1}

    def _calculate_family_size(self, marital_status: str, birth_date: date) -> int:
        """Calculate realistic family size"""
        age = (datetime.now().date() - birth_date).days // 365

        if marital_status == 'Single':
            return 1
        elif marital_status == 'Married':
            if age < 30:
                return random.choices([2, 3, 4], weights=[0.5, 0.3, 0.2])[0]
            elif age < 45:
                return random.choices([2, 3, 4, 5, 6], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]
            else:
                return random.choices([2, 3, 4, 5], weights=[0.3, 0.3, 0.3, 0.1])[0]
        else:  # Divorced, Widowed
            return random.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]

    def _generate_uae_address(self) -> str:
        """Generate realistic UAE address"""
        emirate = random.choice(UAE_EMIRATES)

        if emirate == 'Dubai':
            areas = ['Deira', 'Bur Dubai', 'Jumeirah', 'Marina', 'Downtown', 'Al Karama', 'Satwa']
        elif emirate == 'Abu Dhabi':
            areas = ['Khalifa City', 'Al Reem Island', 'Corniche', 'Al Nahyan', 'Tourist Club']
        else:
            areas = ['City Center', 'Al Nakheel', 'Al Majaz', 'Al Qasimia']

        area = random.choice(areas)
        building = random.randint(1, 999)
        apartment = random.randint(101, 2505)

        return f"Apartment {apartment}, Building {building}, {area}, {emirate}, UAE"

    def generate_bank_statement_data(self, profile_type: str, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic bank statement data with realistic patterns"""
        age = applicant_data['age']
        family_size = applicant_data['family_size']
        education = applicant_data['education_level']
        emirate = applicant_data['emirate']

        # Income ranges based on profile type, education, and location
        income_ranges = self._get_income_range(profile_type, education, emirate, age)
        base_income = random.randint(*income_ranges)

        # Employment status affects income regularity
        employment_status = self._determine_employment_status(profile_type, age)

        # Generate 3 months of transactions for better pattern analysis
        all_transactions = []
        monthly_data = []

        for month in range(3):
            month_start = datetime.now() - timedelta(days=30 * (month + 1))
            month_end = datetime.now() - timedelta(days=30 * month)

            month_income = 0
            month_expenses = 0
            month_transactions = []

            # Income transactions
            if employment_status == 'Employed':
                # Regular salary
                salary = base_income + random.randint(-500, 500)  # Some variation
                month_income += salary
                month_transactions.append({
                    'date': (month_start + timedelta(days=random.randint(25, 30))).strftime('%d/%m/%Y'),
                    'description': f'Salary Transfer - {fake.company()}',
                    'amount': salary,
                    'category': 'Income'
                })
            elif employment_status == 'Self-employed':
                # Irregular income
                for _ in range(random.randint(2, 5)):
                    amount = random.randint(int(base_income * 0.2), int(base_income * 0.6))
                    month_income += amount
                    month_transactions.append({
                        'date': fake.date_between(start_date=month_start, end_date=month_end).strftime('%d/%m/%Y'),
                        'description': random.choice(['Client Payment', 'Service Fee', 'Consultation Fee', 'Project Payment']),
                        'amount': amount,
                        'category': 'Income'
                    })
            elif employment_status == 'Part-time':
                # Lower, irregular income
                for _ in range(random.randint(2, 3)):
                    amount = random.randint(int(base_income * 0.3), int(base_income * 0.5))
                    month_income += amount
                    month_transactions.append({
                        'date': fake.date_between(start_date=month_start, end_date=month_end).strftime('%d/%m/%Y'),
                        'description': f'Part-time Salary - {fake.company()}',
                        'amount': amount,
                        'category': 'Income'
                    })
            elif employment_status == 'Unemployed':
                # Minimal or no income
                if random.random() < 0.3:  # 30% chance of some income (freelance, etc.)
                    amount = random.randint(500, 2000)
                    month_income += amount
                    month_transactions.append({
                        'date': fake.date_between(start_date=month_start, end_date=month_end).strftime('%d/%m/%Y'),
                        'description': random.choice(['Freelance Payment', 'Family Support', 'Odd Job Payment']),
                        'amount': amount,
                        'category': 'Income'
                    })

            # Expense transactions based on family size and income level
            expenses = self._generate_realistic_expenses(month_income, family_size, emirate, month_start, month_end)
            month_transactions.extend(expenses)
            month_expenses = sum(abs(t['amount']) for t in expenses)

            monthly_data.append({
                'month': month + 1,
                'income': month_income,
                'expenses': month_expenses,
                'net': month_income - month_expenses
            })
            all_transactions.extend(month_transactions)

        # Calculate averages
        avg_monthly_income = sum(m['income'] for m in monthly_data) / 3
        avg_monthly_expenses = sum(m['expenses'] for m in monthly_data) / 3

        # Opening balance based on financial stability
        if avg_monthly_income > avg_monthly_expenses:
            opening_balance = random.randint(int(avg_monthly_income * 0.5), int(avg_monthly_income * 2))
        else:
            opening_balance = random.randint(100, int(max(1000, avg_monthly_income * 0.3)))

        # Calculate final balance
        total_net = sum(m['net'] for m in monthly_data)
        closing_balance = opening_balance + total_net

        return {
            'account_number': f"AE{random.randint(10, 99)}{random.randint(1000000000000000, 9999999999999999)}",
            'account_holder': applicant_data['full_name'],
            'bank_name': random.choice(BANKS),
            'statement_period': {
                'start_date': (datetime.now() - timedelta(days=90)).strftime('%d/%m/%Y'),
                'end_date': datetime.now().strftime('%d/%m/%Y')
            },
            'opening_balance': opening_balance,
            'closing_balance': closing_balance,
            'transactions': sorted(all_transactions, key=lambda x: datetime.strptime(x['date'], '%d/%m/%Y')),
            'monthly_income': avg_monthly_income,
            'monthly_expenses': avg_monthly_expenses,
            'monthly_net': avg_monthly_income - avg_monthly_expenses,
            'average_balance': (opening_balance + closing_balance) / 2,
            'employment_status': employment_status,
            'monthly_breakdown': monthly_data,
            'debt_to_income_ratio': min(avg_monthly_expenses / max(avg_monthly_income, 1), 2.0)
        }

    def _get_income_range(self, profile_type: str, education: str, emirate: str, age: int) -> tuple:
        """Get realistic income range based on multiple factors"""
        base_ranges = {
            'eligible': (2000, 12000),
            'borderline': (8000, 18000),
            'high_income': (18000, 60000),
            'unemployed': (0, 3000)
        }

        min_income, max_income = base_ranges.get(profile_type, (2000, 15000))

        # Education multiplier
        education_multipliers = {
            'High School': 0.8,
            'Vocational Training': 0.9,
            'Technical Certificate': 0.95,
            'Diploma': 1.0,
            'Bachelor\'s Degree': 1.3,
            'Master\'s Degree': 1.8,
            'PhD': 2.2
        }
        multiplier = education_multipliers.get(education, 1.0)

        # Emirate multiplier (Dubai/Abu Dhabi typically higher)
        emirate_multipliers = {
            'Dubai': 1.2,
            'Abu Dhabi': 1.15,
            'Sharjah': 1.0,
            'Ajman': 0.9,
            'Ras Al Khaimah': 0.85,
            'Fujairah': 0.8,
            'Umm Al Quwain': 0.8
        }
        location_mult = emirate_multipliers.get(emirate, 1.0)

        # Age multiplier (experience factor)
        if age < 25:
            age_mult = 0.8
        elif age < 35:
            age_mult = 1.0
        elif age < 50:
            age_mult = 1.2
        else:
            age_mult = 1.1  # May be lower due to age discrimination

        final_min = int(min_income * multiplier * location_mult * age_mult)
        final_max = int(max_income * multiplier * location_mult * age_mult)

        return (max(500, final_min), max(1000, final_max))

    def _determine_employment_status(self, profile_type: str, age: int) -> str:
        """Determine employment status based on profile and age"""
        if profile_type == 'unemployed':
            return 'Unemployed'
        elif profile_type == 'high_income':
            return random.choices(['Employed', 'Self-employed'], weights=[0.8, 0.2])[0]
        else:
            if age < 25:
                return random.choices(['Employed', 'Part-time', 'Unemployed'], weights=[0.5, 0.3, 0.2])[0]
            elif age < 60:
                return random.choices(['Employed', 'Self-employed', 'Part-time', 'Unemployed'], weights=[0.6, 0.2, 0.1, 0.1])[0]
            else:
                return random.choices(['Employed', 'Retired', 'Part-time'], weights=[0.3, 0.5, 0.2])[0]

    def _generate_realistic_expenses(self, monthly_income: float, family_size: int, emirate: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Generate realistic expense transactions"""
        expenses = []

        # Rent (biggest expense)
        if emirate in ['Dubai', 'Abu Dhabi']:
            min_rent = 3000
            max_rent = max(min_rent + 100, min(12000, int(monthly_income * 0.4)))
            rent_range = (min_rent, max_rent)
        else:
            min_rent = 2000
            max_rent = max(min_rent + 100, min(8000, int(monthly_income * 0.35)))
            rent_range = (min_rent, max_rent)

        rent = random.randint(*rent_range)
        expenses.append({
            'date': (start_date + timedelta(days=random.randint(1, 5))).strftime('%d/%m/%Y'),
            'description': 'Rent Payment',
            'amount': -rent,
            'category': 'Housing'
        })

        # Utilities (scaled by family size)
        utilities = random.randint(200 * family_size, 500 * family_size)
        expenses.append({
            'date': fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
            'description': 'DEWA/ADDC Bill',
            'amount': -utilities,
            'category': 'Utilities'
        })

        # Groceries (family size dependent)
        grocery_base = 300 + (family_size * 200)
        for _ in range(random.randint(4, 8)):  # Multiple grocery trips
            amount = random.randint(int(grocery_base * 0.2), int(grocery_base * 0.4))
            expenses.append({
                'date': fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
                'description': random.choice(['Carrefour', 'Lulu Hypermarket', 'Spinneys', 'Union Coop', 'Grocery Shopping']),
                'amount': -amount,
                'category': 'Groceries'
            })

        # Transportation
        transport_amount = random.randint(200, 600)
        for _ in range(random.randint(5, 15)):  # Multiple fuel/transport payments
            amount = random.randint(50, 200)
            expenses.append({
                'date': fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
                'description': random.choice(['ADNOC Fuel', 'EPPCO', 'Salik Toll', 'Taxi/Uber', 'Metro Card']),
                'amount': -amount,
                'category': 'Transportation'
            })

        # Communications
        telecom = random.randint(150, 400)
        expenses.append({
            'date': fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
            'description': random.choice(['Etisalat', 'Du', 'Virgin Mobile']),
            'amount': -telecom,
            'category': 'Communications'
        })

        # Healthcare (family dependent)
        if random.random() < 0.7:  # 70% chance of healthcare expenses
            healthcare = random.randint(200 * family_size, 800 * family_size)
            expenses.append({
                'date': fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
                'description': random.choice(['Hospital Bill', 'Pharmacy', 'Medical Center', 'Health Insurance']),
                'amount': -healthcare,
                'category': 'Healthcare'
            })

        # Education (if children)
        if family_size > 2 and random.random() < 0.8:
            education_cost = random.randint(1000, 5000)
            expenses.append({
                'date': fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
                'description': 'School Fees',
                'amount': -education_cost,
                'category': 'Education'
            })

        # Miscellaneous expenses
        misc_categories = [
            ('Restaurant/Dining', 100, 500),
            ('Shopping/Retail', 200, 1000),
            ('Entertainment', 100, 400),
            ('Personal Care', 100, 300)
        ]

        for category, min_amt, max_amt in misc_categories:
            if random.random() < 0.6:  # 60% chance for each category
                amount = random.randint(min_amt, max_amt)
                expenses.append({
                    'date': fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
                    'description': category.split('/')[0],
                    'amount': -amount,
                    'category': category.split('/')[0]
                })

        return expenses

    def generate_resume_data(self, applicant_data: Dict[str, Any], bank_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic resume data aligned with applicant profile"""
        age = applicant_data['age']
        education_level = applicant_data['education_level']
        employment_status = bank_data['employment_status']

        # Calculate realistic experience years
        if education_level in ['High School', 'Vocational Training']:
            max_exp = max(0, age - 18)
        elif education_level in ['Diploma', 'Technical Certificate']:
            max_exp = max(0, age - 20)
        elif education_level == 'Bachelor\'s Degree':
            max_exp = max(0, age - 22)
        elif education_level == 'Master\'s Degree':
            max_exp = max(0, age - 24)
        else:  # PhD
            max_exp = max(0, age - 27)

        # Adjust experience based on employment status
        if employment_status == 'Unemployed':
            experience_years = random.randint(0, max(1, max_exp - 2)) if max_exp > 2 else random.randint(0, max_exp)
        elif employment_status == 'Part-time':
            experience_years = random.randint(0, max(3, max_exp))
        else:
            experience_years = random.randint(0, max_exp) if max_exp > 0 else 0

        # Choose career field based on education and income level
        career_field = self._choose_career_field(education_level, bank_data.get('monthly_income', 5000))

        # Generate skills aligned with career field
        skills = self._generate_aligned_skills(career_field, education_level, experience_years)

        # Generate work experience
        experience_history = self._generate_work_experience(
            career_field, experience_years, employment_status, age
        )

        # Generate certifications
        certifications = self._generate_certifications(career_field, education_level)

        # Current employment status
        current_position = None
        current_company = None
        if employment_status in ['Employed', 'Self-employed']:
            if career_field in JOB_CATEGORIES:
                current_position = random.choice(JOB_CATEGORIES[career_field])
            else:
                current_position = fake.job()
            current_company = fake.company() if employment_status == 'Employed' else 'Self-Employed'

        # Generate education details
        education_details = self._generate_education_details(education_level, age)

        return {
            'name': applicant_data['full_name'],
            'email': applicant_data['email'],
            'phone': applicant_data['phone'],
            'address': applicant_data['address'],
            'career_field': career_field,
            'current_position': current_position,
            'current_company': current_company,
            'employment_status': employment_status,
            'total_experience_years': experience_years,
            'education': education_details,
            'experience': experience_history,
            'skills': skills,
            'certifications': certifications,
            'languages': self._generate_language_skills(applicant_data['nationality']),
            'career_objective': self._generate_career_objective(career_field, experience_years, employment_status),
            'salary_expectation': self._calculate_salary_expectation(bank_data.get('monthly_income', 0), employment_status)
        }

    def _choose_career_field(self, education_level: str, income: float) -> str:
        """Choose appropriate career field based on education and income"""
        if education_level in ['PhD', 'Master\'s Degree']:
            return random.choices(
                ['technology', 'healthcare', 'education', 'business', 'finance'],
                weights=[0.3, 0.2, 0.2, 0.2, 0.1]
            )[0]
        elif education_level == 'Bachelor\'s Degree':
            if income > 20000:
                return random.choices(
                    ['technology', 'business', 'finance', 'healthcare'],
                    weights=[0.35, 0.25, 0.25, 0.15]
                )[0]
            else:
                return random.choices(
                    ['business', 'retail', 'hospitality', 'education'],
                    weights=[0.3, 0.25, 0.25, 0.2]
                )[0]
        elif education_level in ['Diploma', 'Technical Certificate']:
            return random.choices(
                ['technology', 'healthcare', 'construction', 'retail'],
                weights=[0.25, 0.25, 0.25, 0.25]
            )[0]
        else:  # High School, Vocational
            return random.choices(
                ['retail', 'hospitality', 'construction', 'business'],
                weights=[0.3, 0.3, 0.2, 0.2]
            )[0]

    def _generate_aligned_skills(self, career_field: str, education_level: str, experience_years: int) -> List[str]:
        """Generate skills aligned with career field and experience"""
        skills = set()

        # Add field-specific technical skills
        if career_field in SKILL_CATEGORIES['technical']:
            field_skills = SKILL_CATEGORIES['technical'][career_field]
            num_skills = min(len(field_skills), 3 + (experience_years // 2))
            skills.update(random.sample(field_skills, min(num_skills, len(field_skills))))

        # Add general technical skills
        general_tech = SKILL_CATEGORIES['technical']['general']
        skills.update(random.sample(general_tech, min(3, len(general_tech))))

        # Add soft skills based on experience
        soft_skills = SKILL_CATEGORIES['soft']
        num_soft = 3 + (experience_years // 3)
        skills.update(random.sample(soft_skills, min(num_soft, len(soft_skills))))

        # Add language skills
        languages = SKILL_CATEGORIES['languages']
        num_lang = random.randint(2, 4)
        skills.update(random.sample(languages, min(num_lang, len(languages))))

        return list(skills)

    def _generate_work_experience(self, career_field: str, total_years: int, employment_status: str, age: int) -> List[Dict[str, Any]]:
        """Generate realistic work experience"""
        if total_years == 0:
            return []

        experience = []
        years_covered = 0
        current_year = datetime.now().year

        # If currently employed, add current job
        if employment_status in ['Employed', 'Self-employed'] and total_years > 0:
            current_duration = random.randint(1, min(4, total_years))
            start_year = current_year - current_duration

            job_title = random.choice(JOB_CATEGORIES.get(career_field, ['Professional']))
            company = fake.company() if employment_status == 'Employed' else 'Self-Employed'

            experience.append({
                'title': job_title,
                'company': company,
                'start_date': f"{start_year}-01",
                'end_date': 'Present',
                'duration_years': current_duration,
                'responsibilities': self._generate_job_responsibilities(job_title, current_duration),
                'employment_type': employment_status
            })

            years_covered += current_duration

        # Add previous jobs
        while years_covered < total_years:
            remaining_years = total_years - years_covered
            job_duration = random.randint(1, min(5, remaining_years))

            start_year = current_year - years_covered - job_duration
            end_year = current_year - years_covered

            job_title = random.choice(JOB_CATEGORIES.get(career_field, ['Professional']))

            experience.append({
                'title': job_title,
                'company': fake.company(),
                'start_date': f"{start_year}-{random.randint(1, 12):02d}",
                'end_date': f"{end_year}-{random.randint(1, 12):02d}",
                'duration_years': job_duration,
                'responsibilities': self._generate_job_responsibilities(job_title, job_duration),
                'employment_type': 'Employed'
            })

            years_covered += job_duration

        return sorted(experience, key=lambda x: x['start_date'], reverse=True)

    def _generate_job_responsibilities(self, job_title: str, duration: int) -> List[str]:
        """Generate job responsibilities based on title and duration"""
        base_responsibilities = {
            'Software Developer': [
                'Developed and maintained web applications',
                'Collaborated with cross-functional teams',
                'Implemented new features and bug fixes',
                'Participated in code reviews',
                'Optimized application performance'
            ],
            'Sales Manager': [
                'Managed sales team and territory',
                'Developed sales strategies and targets',
                'Built relationships with key clients',
                'Analyzed market trends',
                'Prepared sales reports and forecasts'
            ],
            'Teacher': [
                'Planned and delivered engaging lessons',
                'Assessed student progress and performance',
                'Managed classroom environment',
                'Collaborated with parents and colleagues',
                'Participated in professional development'
            ]
        }

        # Get relevant responsibilities or generate generic ones
        if job_title in base_responsibilities:
            responsibilities = base_responsibilities[job_title][:]
        else:
            responsibilities = [
                'Performed daily operational tasks',
                'Collaborated with team members',
                'Maintained quality standards',
                'Supported organizational objectives'
            ]

        # Add more responsibilities for longer tenures
        if duration >= 3:
            responsibilities.extend([
                'Mentored junior team members',
                'Led special projects and initiatives'
            ])
        if duration >= 5:
            responsibilities.extend([
                'Managed departmental processes',
                'Contributed to strategic planning'
            ])

        return responsibilities[:random.randint(3, len(responsibilities))]

    def _generate_certifications(self, career_field: str, education_level: str) -> List[str]:
        """Generate relevant certifications"""
        cert_mapping = {
            'technology': ['AWS Certified', 'Microsoft Certified', 'CompTIA+', 'Cisco CCNA', 'Google Analytics'],
            'business': ['PMP', 'Six Sigma', 'Digital Marketing', 'CRM Certified', 'Lean Management'],
            'healthcare': ['First Aid Certified', 'Healthcare Administration', 'Medical Coding', 'Patient Care'],
            'finance': ['CFA', 'FRM', 'Accounting Certification', 'Financial Planning', 'Tax Preparation'],
            'education': ['Teaching License', 'TESOL', 'Child Development', 'Educational Technology']
        }

        certs = cert_mapping.get(career_field, ['Professional Development', 'Industry Training'])

        if education_level in ['Master\'s Degree', 'PhD']:
            num_certs = random.randint(2, 4)
        elif education_level == 'Bachelor\'s Degree':
            num_certs = random.randint(1, 3)
        else:
            num_certs = random.randint(0, 2)

        return random.sample(certs, min(num_certs, len(certs)))

    def _generate_language_skills(self, nationality: str) -> List[Dict[str, str]]:
        """Generate language skills based on nationality"""
        languages = []

        # Native language based on nationality
        native_lang_map = {
            'UAE': 'Arabic',
            'Egyptian': 'Arabic',
            'Syrian': 'Arabic',
            'Lebanese': 'Arabic',
            'Jordanian': 'Arabic',
            'Palestinian': 'Arabic',
            'Indian': 'Hindi',
            'Pakistani': 'Urdu',
            'Bangladeshi': 'Bengali',
            'Filipino': 'Tagalog'
        }

        native = native_lang_map.get(nationality, 'English')
        languages.append({'language': native, 'proficiency': 'Native'})

        # English (almost everyone in UAE)
        if native != 'English':
            english_levels = ['Fluent', 'Advanced', 'Intermediate']
            languages.append({'language': 'English', 'proficiency': random.choice(english_levels)})

        # Arabic (for non-Arabic speakers in UAE)
        if native != 'Arabic' and random.random() < 0.7:
            arabic_levels = ['Basic', 'Intermediate', 'Advanced']
            languages.append({'language': 'Arabic', 'proficiency': random.choice(arabic_levels)})

        return languages

    def _generate_education_details(self, education_level: str, age: int) -> List[Dict[str, Any]]:
        """Generate detailed education information"""
        education = []
        current_year = datetime.now().year

        if education_level == 'High School':
            grad_year = current_year - (age - 18)
            education.append({
                'degree': 'High School Diploma',
                'institution': f"{random.choice(UAE_EMIRATES)} Secondary School",
                'graduation_year': grad_year,
                'gpa': round(random.uniform(2.5, 4.0), 2)
            })
        else:
            # High school first
            hs_year = current_year - (age - 18)
            education.append({
                'degree': 'High School Diploma',
                'institution': f"{random.choice(UAE_EMIRATES)} Secondary School",
                'graduation_year': hs_year,
                'gpa': round(random.uniform(2.8, 4.0), 2)
            })

            # Higher education
            if education_level in ['Diploma', 'Technical Certificate']:
                grad_year = current_year - (age - 20)
                education.append({
                    'degree': education_level,
                    'institution': f"{fake.company()} Technical Institute",
                    'graduation_year': grad_year,
                    'gpa': round(random.uniform(3.0, 4.0), 2)
                })
            elif education_level == 'Bachelor\'s Degree':
                grad_year = current_year - (age - 22)
                education.append({
                    'degree': f"Bachelor's in {random.choice(['Business', 'Engineering', 'Computer Science', 'Management'])}",
                    'institution': f"{random.choice(['American University', 'UAE University', 'Zayed University'])}",
                    'graduation_year': grad_year,
                    'gpa': round(random.uniform(2.8, 3.8), 2)
                })
            elif education_level == 'Master\'s Degree':
                # Bachelor's first
                bachelor_year = current_year - (age - 22)
                education.append({
                    'degree': f"Bachelor's in {random.choice(['Business', 'Engineering', 'Computer Science'])}",
                    'institution': f"{random.choice(['American University', 'UAE University'])}",
                    'graduation_year': bachelor_year,
                    'gpa': round(random.uniform(3.0, 3.8), 2)
                })

                # Master's
                masters_year = current_year - (age - 24)
                education.append({
                    'degree': f"Master's in {random.choice(['Business Administration', 'Engineering Management', 'Computer Science'])}",
                    'institution': f"{random.choice(['American University', 'INSEAD', 'London Business School'])}",
                    'graduation_year': masters_year,
                    'gpa': round(random.uniform(3.2, 4.0), 2)
                })

        return education

    def _generate_career_objective(self, career_field: str, experience_years: int, employment_status: str) -> str:
        """Generate career objective statement"""
        objectives = {
            'technology': [
                "Seeking opportunities to leverage technical skills in software development",
                "Passionate about creating innovative technology solutions",
                "Looking to contribute to digital transformation initiatives"
            ],
            'business': [
                "Dedicated professional seeking to drive business growth and efficiency",
                "Committed to delivering exceptional results in dynamic business environments",
                "Seeking opportunities to apply analytical and leadership skills"
            ],
            'healthcare': [
                "Compassionate healthcare professional dedicated to patient care",
                "Committed to improving healthcare outcomes through quality service",
                "Seeking to contribute to healthcare excellence and innovation"
            ]
        }

        field_objectives = objectives.get(career_field, [
            "Motivated professional seeking growth opportunities",
            "Dedicated to contributing value and achieving organizational goals"
        ])

        base_objective = random.choice(field_objectives)

        if employment_status == 'Unemployed':
            return f"{base_objective} and secure stable employment in a challenging role."
        elif experience_years < 3:
            return f"{base_objective} while developing professional expertise and skills."
        else:
            return f"{base_objective} and advance to senior leadership positions."

    def _calculate_salary_expectation(self, current_income: float, employment_status: str) -> Dict[str, float]:
        """Calculate realistic salary expectations"""
        if employment_status == 'Unemployed':
            return {
                'minimum': max(3000, current_income * 0.8),
                'preferred': max(5000, current_income * 1.2),
                'maximum': max(8000, current_income * 1.5)
            }
        else:
            return {
                'minimum': current_income * 1.1,
                'preferred': current_income * 1.3,
                'maximum': current_income * 1.6
            }

    def generate_assets_liabilities_data(self, wealth_level: str = 'medium') -> Dict[str, Any]:
        """Generate synthetic assets and liabilities data"""
        wealth_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.5
        }

        multiplier = wealth_multipliers[wealth_level]

        # Assets
        assets = {
            'Cash and Bank Accounts': random.randint(5000, 50000) * multiplier,
            'Investment Accounts': random.randint(0, 100000) * multiplier,
            'Property/Real Estate': random.randint(0, 1000000) * multiplier if random.random() > 0.6 else 0,
            'Vehicle': random.randint(20000, 150000) * multiplier if random.random() > 0.4 else 0,
            'Personal Items': random.randint(10000, 50000) * multiplier,
            'Other Assets': random.randint(0, 25000) * multiplier if random.random() > 0.7 else 0
        }

        # Liabilities
        liabilities = {
            'Credit Card Debt': random.randint(0, 25000) * multiplier if random.random() > 0.5 else 0,
            'Personal Loan': random.randint(0, 100000) * multiplier if random.random() > 0.6 else 0,
            'Car Loan': random.randint(0, 80000) * multiplier if assets['Vehicle'] > 0 else 0,
            'Mortgage': random.randint(0, 800000) * multiplier if assets['Property/Real Estate'] > 0 else 0,
            'Other Debts': random.randint(0, 15000) * multiplier if random.random() > 0.8 else 0
        }

        # Remove zero values
        assets = {k: v for k, v in assets.items() if v > 0}
        liabilities = {k: v for k, v in liabilities.items() if v > 0}

        total_assets = sum(assets.values())
        total_liabilities = sum(liabilities.values())

        # Calculate liquid vs fixed assets
        liquid_assets = assets.get('Cash and Bank Accounts', 0) + assets.get('Investment Accounts', 0)
        fixed_assets = total_assets - liquid_assets

        return {
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'net_worth': total_assets - total_liabilities,
            'assets_breakdown': assets,
            'liabilities_breakdown': liabilities,
            'liquid_assets': liquid_assets,
            'fixed_assets': fixed_assets
        }

    def generate_credit_report_data(self, income_level: str = 'medium') -> Dict[str, Any]:
        """Generate synthetic credit report data"""
        # Credit score ranges based on income level
        score_ranges = {
            'low': (500, 650),
            'medium': (600, 750),
            'high': (700, 850)
        }

        credit_score = random.randint(*score_ranges[income_level])

        # Credit rating based on score
        if credit_score >= 800:
            rating = 'Excellent'
        elif credit_score >= 750:
            rating = 'Very Good'
        elif credit_score >= 700:
            rating = 'Good'
        elif credit_score >= 650:
            rating = 'Fair'
        else:
            rating = 'Poor'

        # Other credit details
        total_debt = random.randint(0, 200000)
        monthly_obligations = total_debt * random.uniform(0.02, 0.08)  # 2-8% of total debt
        credit_utilization = random.randint(10, 85)
        payment_history = random.randint(70, 100)

        return {
            'credit_score': credit_score,
            'credit_rating': rating,
            'total_debt': total_debt,
            'monthly_obligations': monthly_obligations,
            'credit_utilization': credit_utilization,
            'payment_history': payment_history,
            'accounts': [
                {
                    'type': 'Credit Card',
                    'balance': random.randint(0, 25000),
                    'limit': random.randint(10000, 50000),
                    'status': random.choice(['Current', 'Late 30 days', 'Current'])
                },
                {
                    'type': 'Personal Loan',
                    'balance': random.randint(0, 75000),
                    'original_amount': random.randint(50000, 100000),
                    'status': random.choice(['Current', 'Current', 'Late 30 days'])
                }
            ],
            'inquiries': random.randint(0, 5),
            'defaults': [] if credit_score > 600 else [{'amount': random.randint(5000, 25000), 'date': '2023-03-15'}]
        }

    def generate_complete_application(self, profile_type: str = 'eligible') -> Dict[str, Any]:
        """Generate a complete application with all documents"""

        # Define profile characteristics
        profiles = {
            'eligible': {
                'income_level': 'low',
                'wealth_level': 'low',
                'experience_years': random.randint(1, 15),
                'education': random.choice(['High School', 'Diploma', 'Bachelor\'s Degree'])
            },
            'high_income': {
                'income_level': 'high',
                'wealth_level': 'high',
                'experience_years': random.randint(5, 20),
                'education': random.choice(['Bachelor\'s Degree', 'Master\'s Degree', 'PhD'])
            },
            'borderline': {
                'income_level': 'medium',
                'wealth_level': 'medium',
                'experience_years': random.randint(2, 10),
                'education': random.choice(['Diploma', 'Bachelor\'s Degree'])
            }
        }

        profile = profiles.get(profile_type, profiles['eligible'])

        # Generate all components
        applicant = self.generate_applicant_data(profile_type)
        bank_statement = self.generate_bank_statement_data(profile_type, applicant)
        resume = self.generate_resume_data(applicant, bank_statement)

        # Determine wealth level based on income and age
        monthly_income = bank_statement['monthly_income']
        if monthly_income < 8000:
            wealth_level = 'low'
        elif monthly_income < 20000:
            wealth_level = 'medium'
        else:
            wealth_level = 'high'

        assets_liabilities = self.generate_assets_liabilities_data(wealth_level)

        # Determine income level for credit report
        if monthly_income < 8000:
            income_level = 'low'
        elif monthly_income < 20000:
            income_level = 'medium'
        else:
            income_level = 'high'

        credit_report = self.generate_credit_report_data(income_level)

        # Determine support type based on profile
        if profile_type in ['unemployed', 'eligible'] and monthly_income < 15000:
            support_type = random.choices(
                ['financial', 'economic_enablement', 'both'],
                weights=[0.4, 0.3, 0.3]
            )[0]
        elif profile_type == 'borderline':
            support_type = random.choices(
                ['financial', 'economic_enablement', 'both'],
                weights=[0.2, 0.5, 0.3]
            )[0]
        else:
            support_type = 'economic_enablement'  # High income only eligible for skills development

        # Calculate expected outcome for training data
        expected_outcome = self._calculate_expected_outcome(applicant, bank_statement, assets_liabilities, credit_report)

        return {
            'application_id': None,  # Will be set later
            'applicant': applicant,
            'support_type': support_type,
            'documents': {
                'emirates_id': {
                    'id_number': applicant['emirates_id'],
                    'name': applicant['full_name'],
                    'nationality': applicant['nationality'],
                    'date_of_birth': applicant['date_of_birth'],
                    'gender': applicant['gender'],
                    'address': applicant['address'],
                    'issue_date': '15/03/2020',
                    'expiry_date': '15/03/2030',
                    'issuing_authority': 'Federal Authority for Identity and Citizenship'
                },
                'bank_statement': bank_statement,
                'resume': resume,
                'assets_liabilities': assets_liabilities,
                'credit_report': credit_report
            },
            'profile_type': profile_type,
            'expected_outcome': expected_outcome,
            'submission_date': fake.date_between(start_date='-60d', end_date='today').isoformat(),
            'application_status': 'pending'
        }

    def _calculate_expected_outcome(self, applicant: Dict, bank_data: Dict, assets: Dict, credit: Dict) -> Dict[str, Any]:
        """Calculate expected eligibility outcome for training purposes"""
        age = applicant['age']
        monthly_income = bank_data['monthly_income']
        net_worth = assets['net_worth']
        debt_to_income = bank_data.get('debt_to_income_ratio', 0.5)
        family_size = applicant['family_size']
        employment_status = bank_data['employment_status']

        # Calculate eligibility score based on criteria
        score = 0.5  # Base score
        reasons = []

        # Age criteria (18-65)
        if 18 <= age <= 65:
            score += 0.1
        else:
            score -= 0.2
            reasons.append(f"Age ({age}) outside eligible range (18-65)")

        # Income criteria
        if monthly_income <= 15000:  # Financial support threshold
            score += 0.2
            if monthly_income < 5000:
                score += 0.1  # Very low income gets bonus
        else:
            if monthly_income <= 25000:  # Economic enablement threshold
                score += 0.1
            else:
                score -= 0.2
                reasons.append(f"Income ({monthly_income}) exceeds maximum threshold")

        # Net worth criteria
        if net_worth <= 500000:
            score += 0.1
        else:
            score -= 0.2
            reasons.append(f"Net worth ({net_worth}) exceeds limit")

        # Debt to income ratio
        if debt_to_income <= 0.6:
            score += 0.1
        else:
            score -= 0.15
            reasons.append(f"Debt-to-income ratio ({debt_to_income:.2f}) too high")

        # Family size (need factor)
        if family_size > 2:
            score += 0.05 * (family_size - 2)  # Bonus for dependents

        # Employment status
        if employment_status == 'Unemployed':
            score += 0.15  # Higher need
        elif employment_status == 'Part-time':
            score += 0.1
        elif employment_status == 'Employed':
            score += 0.05

        # Credit score factor
        credit_score = credit.get('credit_score', 650)
        if credit_score >= 700:
            score += 0.05
        elif credit_score < 600:
            score -= 0.05

        # Normalize score between 0 and 1
        final_score = max(0, min(1, score))

        # Determine expected decision
        if final_score >= 0.85:
            decision = 'approve'
        elif final_score >= 0.65:
            decision = 'conditional_approve'
        elif final_score >= 0.45:
            decision = 'manual_review'
        else:
            decision = 'decline'

        # Check for critical failures
        if age < 18 or age > 65:
            decision = 'decline'
        if monthly_income > 25000:
            decision = 'decline'
        if net_worth > 500000:
            decision = 'decline'

        return {
            'eligibility_score': round(final_score, 3),
            'expected_decision': decision,
            'reasons': reasons,
            'risk_level': 'Low' if final_score > 0.7 else 'Medium' if final_score > 0.4 else 'High'
        }

    def generate_test_dataset(self, num_applications: int = 50) -> List[Dict[str, Any]]:
        """Generate a complete test dataset"""
        applications = []

        # Distribution of profile types
        profile_distribution = {
            'eligible': 0.6,      # 60% eligible
            'borderline': 0.25,   # 25% borderline
            'high_income': 0.15   # 15% high income (likely ineligible)
        }

        for i in range(num_applications):
            # Choose profile type based on distribution
            rand = random.random()
            if rand < profile_distribution['eligible']:
                profile_type = 'eligible'
            elif rand < profile_distribution['eligible'] + profile_distribution['borderline']:
                profile_type = 'borderline'
            else:
                profile_type = 'high_income'

            application = self.generate_complete_application(profile_type)
            application['application_id'] = f"APP-2024-{str(i+1).zfill(4)}"
            applications.append(application)

        return applications

    def save_test_data(self, num_applications: int = 50):
        """Generate and save test data to files"""
        print(f"Generating {num_applications} synthetic applications...")

        # Generate dataset
        applications = self.generate_test_dataset(num_applications)

        # Save complete dataset
        with open(self.output_dir / 'complete_applications.json', 'w') as f:
            json.dump(applications, f, indent=2, default=str)

        # Save individual components for easier access
        applicants = []
        documents = {}

        for app in applications:
            applicants.append({
                'application_id': app['application_id'],
                **app['applicant'],
                'support_type': app['support_type'],
                'profile_type': app['profile_type']
            })

            for doc_type, doc_data in app['documents'].items():
                if doc_type not in documents:
                    documents[doc_type] = []
                documents[doc_type].append({
                    'application_id': app['application_id'],
                    **doc_data
                })

        # Save applicants data
        pd.DataFrame(applicants).to_csv(self.output_dir / 'applicants.csv', index=False)

        # Save document data
        for doc_type, doc_list in documents.items():
            pd.DataFrame(doc_list).to_csv(self.output_dir / f'{doc_type}_documents.csv', index=False)

        # Save summary statistics
        stats = {
            'total_applications': len(applications),
            'profile_distribution': {
                profile: len([app for app in applications if app['profile_type'] == profile])
                for profile in ['eligible', 'borderline', 'high_income']
            },
            'support_type_distribution': {
                support_type: len([app for app in applications if app['support_type'] == support_type])
                for support_type in ['financial', 'economic_enablement', 'both']
            },
            'generation_date': datetime.now().isoformat()
        }

        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print("Synthetic data generated successfully!")
        print(f"Files saved to: {self.output_dir}")
        print(f"Applications by profile type:")
        for profile, count in stats['profile_distribution'].items():
            print(f"   - {profile}: {count}")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.save_test_data(num_applications=100)