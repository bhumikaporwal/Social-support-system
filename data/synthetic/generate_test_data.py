import json
import random
import pandas as pd
from datetime import datetime, timedelta, date
from faker import Faker
from typing import List, Dict, Any
import uuid
import os
from pathlib import Path

# Initialize Faker with Arabic locale for UAE context
fake = Faker(['ar_AE', 'en_US'])

# UAE-specific data
UAE_EMIRATES = ['Abu Dhabi', 'Dubai', 'Sharjah', 'Ajman', 'Umm Al Quwain', 'Ras Al Khaimah', 'Fujairah']
UAE_NATIONALITIES = ['UAE', 'Egyptian', 'Indian', 'Pakistani', 'Bangladeshi', 'Filipino', 'Syrian', 'Lebanese', 'Jordanian', 'Palestinian']
BANKS = ['Emirates NBD', 'ADCB', 'FAB', 'Mashreq Bank', 'CBD', 'HSBC UAE', 'Citibank UAE']
EDUCATION_LEVELS = ['High School', 'Diploma', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD', 'Vocational Training']
EMPLOYMENT_STATUS = ['Employed', 'Unemployed', 'Self-employed', 'Part-time', 'Contract Worker', 'Retired']
MARITAL_STATUS = ['Single', 'Married', 'Divorced', 'Widowed']

class SyntheticDataGenerator:
    """Generate synthetic test data for the social support system"""

    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_emirates_id(self) -> str:
        """Generate a realistic Emirates ID format"""
        # Format: 784-YYYY-NNNNNNN-N
        birth_year = random.randint(1960, 2005)
        sequence = random.randint(1000000, 9999999)
        check_digit = random.randint(1, 9)
        return f"784-{birth_year}-{sequence}-{check_digit}"

    def generate_applicant_data(self) -> Dict[str, Any]:
        """Generate synthetic applicant data"""
        gender = random.choice(['Male', 'Female'])

        # Generate Arabic name for UAE nationals, mixed for others
        nationality = random.choice(UAE_NATIONALITIES)
        if nationality == 'UAE':
            first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
            last_name = fake.last_name()
        else:
            first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
            last_name = fake.last_name()

        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=70)

        return {
            'emirates_id': self.generate_emirates_id(),
            'first_name': first_name,
            'last_name': last_name,
            'date_of_birth': birth_date.isoformat(),
            'nationality': nationality,
            'gender': gender,
            'email': fake.email(),
            'phone': f"+971{random.randint(50, 58)}{random.randint(1000000, 9999999)}",
            'address': fake.address(),
            'emirate': random.choice(UAE_EMIRATES),
            'education_level': random.choice(EDUCATION_LEVELS),
            'marital_status': random.choice(MARITAL_STATUS),
            'profession': fake.job() if random.random() > 0.1 else None
        }

    def generate_bank_statement_data(self, income_level: str = 'medium') -> Dict[str, Any]:
        """Generate synthetic bank statement data"""
        income_ranges = {
            'low': (2000, 8000),
            'medium': (8000, 20000),
            'high': (20000, 50000)
        }

        base_income = random.randint(*income_ranges[income_level])

        # Generate transactions
        transactions = []
        monthly_income = 0
        monthly_expenses = 0

        # Salary transactions
        for _ in range(random.randint(1, 2)):  # 1-2 salary payments
            amount = base_income / random.randint(1, 2)
            monthly_income += amount
            transactions.append({
                'date': fake.date_between(start_date='-30d', end_date='-1d').strftime('%d/%m/%Y'),
                'description': f'Salary Transfer - {fake.company()}',
                'amount': amount
            })

        # Expense transactions
        expense_categories = [
            ('Grocery Shopping', 200, 800),
            ('Utilities Payment', 300, 600),
            ('Rent Payment', 2000, 8000),
            ('Fuel/Petrol', 150, 400),
            ('Mobile/Internet', 100, 300),
            ('Restaurant/Dining', 50, 200),
            ('Shopping', 100, 500),
            ('Healthcare', 200, 1000)
        ]

        for category, min_amt, max_amt in expense_categories:
            if random.random() > 0.3:  # 70% chance for each expense
                amount = random.randint(min_amt, max_amt)
                monthly_expenses += amount
                transactions.append({
                    'date': fake.date_between(start_date='-30d', end_date='-1d').strftime('%d/%m/%Y'),
                    'description': category,
                    'amount': -amount
                })

        opening_balance = random.randint(1000, 20000)
        closing_balance = opening_balance + monthly_income - monthly_expenses

        return {
            'account_number': f"AE{random.randint(10, 99)}{random.randint(1000000000000000, 9999999999999999)}",
            'account_holder': fake.name(),
            'bank_name': random.choice(BANKS),
            'statement_period': {
                'start_date': (datetime.now() - timedelta(days=30)).strftime('%d/%m/%Y'),
                'end_date': datetime.now().strftime('%d/%m/%Y')
            },
            'opening_balance': opening_balance,
            'closing_balance': closing_balance,
            'transactions': transactions,
            'monthly_income': monthly_income,
            'monthly_expenses': monthly_expenses,
            'average_balance': (opening_balance + closing_balance) / 2
        }

    def generate_resume_data(self, education_level: str, experience_years: int = None) -> Dict[str, Any]:
        """Generate synthetic resume data"""
        if experience_years is None:
            experience_years = random.randint(0, 25)

        skills = []
        skill_categories = {
            'technical': ['Python', 'JavaScript', 'SQL', 'Excel', 'PowerPoint', 'Microsoft Office', 'Data Analysis'],
            'soft': ['Communication', 'Leadership', 'Teamwork', 'Problem Solving', 'Time Management', 'Project Management'],
            'languages': ['Arabic', 'English', 'Hindi', 'Urdu', 'French', 'German']
        }

        # Add random skills
        for category, skill_list in skill_categories.items():
            num_skills = random.randint(1, min(4, len(skill_list)))
            skills.extend(random.sample(skill_list, num_skills))

        # Generate experience
        experience = []
        total_years = 0
        while total_years < experience_years:
            job_years = random.randint(1, min(5, experience_years - total_years))
            start_year = 2024 - experience_years + total_years
            end_year = start_year + job_years

            experience.append(f"{start_year}-{end_year}: {fake.job()} at {fake.company()}")
            total_years += job_years

        return {
            'name': fake.name(),
            'email': fake.email(),
            'phone': f"+971{random.randint(50, 58)}{random.randint(1000000, 9999999)}",
            'education': [f"{education_level} - {fake.company()} University (2015-2019)"],
            'experience': experience,
            'skills': skills,
            'current_position': fake.job() if experience_years > 0 else None,
            'total_experience_years': experience_years
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
        applicant = self.generate_applicant_data()
        bank_statement = self.generate_bank_statement_data(profile['income_level'])
        resume = self.generate_resume_data(profile['education'], profile['experience_years'])
        assets_liabilities = self.generate_assets_liabilities_data(profile['wealth_level'])
        credit_report = self.generate_credit_report_data(profile['income_level'])

        return {
            'applicant': applicant,
            'support_type': random.choice(['financial', 'economic_enablement', 'both']),
            'documents': {
                'emirates_id': {
                    'id_number': applicant['emirates_id'],
                    'name': f"{applicant['first_name']} {applicant['last_name']}",
                    'nationality': applicant['nationality'],
                    'date_of_birth': applicant['date_of_birth'],
                    'gender': applicant['gender'],
                    'issue_date': '15/03/2020',
                    'expiry_date': '15/03/2030'
                },
                'bank_statement': bank_statement,
                'resume': resume,
                'assets_liabilities': assets_liabilities,
                'credit_report': credit_report
            },
            'profile_type': profile_type
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

        print(f"✅ Synthetic data generated successfully!")
        print(f"📁 Files saved to: {self.output_dir}")
        print(f"📊 Applications by profile type:")
        for profile, count in stats['profile_distribution'].items():
            print(f"   - {profile}: {count}")

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.save_test_data(num_applications=100)