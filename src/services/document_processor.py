import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import cv2
import PyPDF2
import openpyxl
from docx import Document as DocxDocument
import re
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    document_type: str
    extracted_data: Dict[str, Any]
    extracted_text: str
    confidence_score: float
    processing_status: str
    errors: List[str]

class DocumentProcessor:
    """Multimodal document processing for various document types"""

    def __init__(self):
        self.supported_types = {
            'bank_statement': self._process_bank_statement,
            'emirates_id': self._process_emirates_id,
            'resume': self._process_resume,
            'assets_liabilities': self._process_assets_liabilities,
            'credit_report': self._process_credit_report
        }

    def process_document(self, file_path: str, document_type: str) -> ExtractionResult:
        """Main entry point for document processing"""
        try:
            if document_type not in self.supported_types:
                raise ValueError(f"Unsupported document type: {document_type}")

            # Extract text based on file type
            extracted_text = self._extract_text_from_file(file_path)

            # Process specific document type
            processor = self.supported_types[document_type]
            extracted_data = processor(file_path, extracted_text)

            # Calculate confidence score
            confidence_score = self._calculate_confidence(extracted_data, document_type)

            return ExtractionResult(
                document_type=document_type,
                extracted_data=extracted_data,
                extracted_text=extracted_text,
                confidence_score=confidence_score,
                processing_status="completed",
                errors=[]
            )

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return ExtractionResult(
                document_type=document_type,
                extracted_data={},
                extracted_text="",
                confidence_score=0.0,
                processing_status="failed",
                errors=[str(e)]
            )

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                return self._extract_text_from_image(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return self._extract_text_from_excel(file_path)
            elif file_extension == '.docx':
                return self._extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
        return text

    def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Load and preprocess image
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply image enhancement
            denoised = cv2.fastNlMeansDenoising(gray)

            # Extract text using Tesseract
            text = pytesseract.image_to_string(denoised, config='--psm 6')
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""

    def _extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"Sheet: {sheet_name}\n"
                text += sheet_df.to_string() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from Excel: {str(e)}")
            return ""

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            return ""

    def _process_bank_statement(self, file_path: str, text: str) -> Dict[str, Any]:
        """Extract structured data from bank statement"""
        data = {
            'account_number': None,
            'account_holder': None,
            'bank_name': None,
            'statement_period': {},
            'opening_balance': None,
            'closing_balance': None,
            'transactions': [],
            'monthly_income': None,
            'monthly_expenses': None,
            'average_balance': None
        }

        try:
            # Extract account number
            account_match = re.search(r'Account\s*(?:Number|No\.?):?\s*(\d+)', text, re.IGNORECASE)
            if account_match:
                data['account_number'] = account_match.group(1)

            # Extract account holder name
            name_patterns = [
                r'Account\s*Holder:?\s*([A-Za-z\s]+)',
                r'Name:?\s*([A-Za-z\s]+)',
                r'Customer:?\s*([A-Za-z\s]+)'
            ]
            for pattern in name_patterns:
                name_match = re.search(pattern, text, re.IGNORECASE)
                if name_match:
                    data['account_holder'] = name_match.group(1).strip()
                    break

            # Extract bank name
            bank_patterns = [
                r'(Emirates NBD|ADCB|FAB|Mashreq|CBD|HSBC|Citibank)',
                r'Bank:?\s*([A-Za-z\s]+Bank)'
            ]
            for pattern in bank_patterns:
                bank_match = re.search(pattern, text, re.IGNORECASE)
                if bank_match:
                    data['bank_name'] = bank_match.group(1).strip()
                    break

            # Extract balances
            balance_patterns = [
                r'Opening\s*Balance:?\s*AED\s*([\d,]+\.?\d*)',
                r'Closing\s*Balance:?\s*AED\s*([\d,]+\.?\d*)',
                r'Balance:?\s*AED\s*([\d,]+\.?\d*)'
            ]

            opening_match = re.search(balance_patterns[0], text, re.IGNORECASE)
            if opening_match:
                data['opening_balance'] = float(opening_match.group(1).replace(',', ''))

            closing_match = re.search(balance_patterns[1], text, re.IGNORECASE)
            if closing_match:
                data['closing_balance'] = float(closing_match.group(1).replace(',', ''))

            # Extract transactions and calculate income/expenses
            transaction_pattern = r'(\d{2}/\d{2}/\d{4})\s+([A-Za-z\s]+)\s+([\-\+]?[\d,]+\.?\d*)'
            transactions = re.findall(transaction_pattern, text)

            total_income = 0
            total_expenses = 0

            for date, description, amount in transactions:
                amount_val = float(amount.replace(',', '').replace('+', ''))
                transaction = {
                    'date': date,
                    'description': description.strip(),
                    'amount': amount_val
                }
                data['transactions'].append(transaction)

                if amount_val > 0:
                    total_income += amount_val
                else:
                    total_expenses += abs(amount_val)

            data['monthly_income'] = total_income
            data['monthly_expenses'] = total_expenses

            if data['opening_balance'] and data['closing_balance']:
                data['average_balance'] = (data['opening_balance'] + data['closing_balance']) / 2

        except Exception as e:
            logger.error(f"Error processing bank statement: {str(e)}")

        return data

    def _process_emirates_id(self, file_path: str, text: str) -> Dict[str, Any]:
        """Extract structured data from Emirates ID"""
        data = {
            'id_number': None,
            'name': None,
            'nationality': None,
            'date_of_birth': None,
            'gender': None,
            'issue_date': None,
            'expiry_date': None
        }

        try:
            # Extract ID number
            id_patterns = [
                r'(\d{3}-\d{4}-\d{7}-\d{1})',
                r'ID\s*(?:Number|No\.?):?\s*(\d{15})',
                r'(\d{15})'
            ]
            for pattern in id_patterns:
                id_match = re.search(pattern, text)
                if id_match:
                    data['id_number'] = id_match.group(1)
                    break

            # Extract name
            name_patterns = [
                r'Name:?\s*([A-Za-z\s]+)',
                r'الاسم:?\s*([A-Za-z\s]+)'
            ]
            for pattern in name_patterns:
                name_match = re.search(pattern, text, re.IGNORECASE)
                if name_match:
                    data['name'] = name_match.group(1).strip()
                    break

            # Extract nationality
            nationality_match = re.search(r'Nationality:?\s*([A-Za-z\s]+)', text, re.IGNORECASE)
            if nationality_match:
                data['nationality'] = nationality_match.group(1).strip()

            # Extract dates
            date_patterns = [
                r'Date\s*of\s*Birth:?\s*(\d{2}/\d{2}/\d{4})',
                r'Issue\s*Date:?\s*(\d{2}/\d{2}/\d{4})',
                r'Expiry\s*Date:?\s*(\d{2}/\d{2}/\d{4})'
            ]

            dob_match = re.search(date_patterns[0], text, re.IGNORECASE)
            if dob_match:
                data['date_of_birth'] = dob_match.group(1)

            issue_match = re.search(date_patterns[1], text, re.IGNORECASE)
            if issue_match:
                data['issue_date'] = issue_match.group(1)

            expiry_match = re.search(date_patterns[2], text, re.IGNORECASE)
            if expiry_match:
                data['expiry_date'] = expiry_match.group(1)

            # Extract gender
            gender_match = re.search(r'(?:Gender|Sex):?\s*(Male|Female|M|F)', text, re.IGNORECASE)
            if gender_match:
                gender = gender_match.group(1).upper()
                data['gender'] = 'Male' if gender in ['MALE', 'M'] else 'Female'

        except Exception as e:
            logger.error(f"Error processing Emirates ID: {str(e)}")

        return data

    def _process_resume(self, file_path: str, text: str) -> Dict[str, Any]:
        """Extract structured data from resume"""
        data = {
            'name': None,
            'email': None,
            'phone': None,
            'education': [],
            'experience': [],
            'skills': [],
            'current_position': None,
            'total_experience_years': None
        }

        try:
            # Extract email
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if email_match:
                data['email'] = email_match.group()

            # Extract phone
            phone_patterns = [
                r'\+971\s*\d{1,2}\s*\d{3}\s*\d{4}',
                r'05\d\s*\d{3}\s*\d{4}',
                r'\d{3}-\d{3}-\d{4}'
            ]
            for pattern in phone_patterns:
                phone_match = re.search(pattern, text)
                if phone_match:
                    data['phone'] = phone_match.group()
                    break

            # Extract name (usually first line or after certain keywords)
            lines = text.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                if re.match(r'^[A-Za-z\s]{2,50}$', line.strip()) and len(line.strip().split()) >= 2:
                    data['name'] = line.strip()
                    break

            # Extract education
            education_section = re.search(r'(?:EDUCATION|Education)(.*?)(?=EXPERIENCE|Experience|SKILLS|Skills|$)', text, re.DOTALL | re.IGNORECASE)
            if education_section:
                education_text = education_section.group(1)
                education_lines = [line.strip() for line in education_text.split('\n') if line.strip()]
                data['education'] = education_lines

            # Extract experience
            experience_section = re.search(r'(?:EXPERIENCE|Experience|WORK|Work)(.*?)(?=EDUCATION|Education|SKILLS|Skills|$)', text, re.DOTALL | re.IGNORECASE)
            if experience_section:
                experience_text = experience_section.group(1)
                # Parse years of experience
                year_matches = re.findall(r'(\d{4})\s*[-–]\s*(\d{4}|Present|Current)', experience_text, re.IGNORECASE)
                total_years = 0
                for start, end in year_matches:
                    end_year = 2024 if end.lower() in ['present', 'current'] else int(end)
                    total_years += end_year - int(start)
                data['total_experience_years'] = total_years

            # Extract skills
            skills_section = re.search(r'(?:SKILLS|Skills)(.*?)(?=EDUCATION|Education|EXPERIENCE|Experience|$)', text, re.DOTALL | re.IGNORECASE)
            if skills_section:
                skills_text = skills_section.group(1)
                skills = [skill.strip() for skill in re.split(r'[,\n•]', skills_text) if skill.strip()]
                data['skills'] = skills

        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")

        return data

    def _process_assets_liabilities(self, file_path: str, text: str) -> Dict[str, Any]:
        """Extract structured data from assets/liabilities Excel file"""
        data = {
            'total_assets': None,
            'total_liabilities': None,
            'net_worth': None,
            'assets_breakdown': {},
            'liabilities_breakdown': {},
            'liquid_assets': None,
            'fixed_assets': None
        }

        try:
            if Path(file_path).suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, sheet_name=None)

                for sheet_name, sheet_df in df.items():
                    # Look for assets and liabilities data
                    if 'asset' in sheet_name.lower():
                        assets_df = sheet_df.dropna()
                        if len(assets_df.columns) >= 2:
                            asset_dict = dict(zip(assets_df.iloc[:, 0], assets_df.iloc[:, 1]))
                            data['assets_breakdown'] = {str(k): float(v) if pd.notna(v) and str(v).replace('.', '').replace(',', '').isdigit() else 0 for k, v in asset_dict.items()}
                            data['total_assets'] = sum(data['assets_breakdown'].values())

                    elif 'liabilit' in sheet_name.lower():
                        liabilities_df = sheet_df.dropna()
                        if len(liabilities_df.columns) >= 2:
                            liability_dict = dict(zip(liabilities_df.iloc[:, 0], liabilities_df.iloc[:, 1]))
                            data['liabilities_breakdown'] = {str(k): float(v) if pd.notna(v) and str(v).replace('.', '').replace(',', '').isdigit() else 0 for k, v in liability_dict.items()}
                            data['total_liabilities'] = sum(data['liabilities_breakdown'].values())

                # Calculate net worth
                if data['total_assets'] and data['total_liabilities']:
                    data['net_worth'] = data['total_assets'] - data['total_liabilities']

                # Categorize liquid vs fixed assets
                liquid_keywords = ['cash', 'bank', 'savings', 'current']
                fixed_keywords = ['property', 'real estate', 'vehicle', 'investment']

                liquid_assets = 0
                fixed_assets = 0

                for asset, value in data['assets_breakdown'].items():
                    asset_lower = asset.lower()
                    if any(keyword in asset_lower for keyword in liquid_keywords):
                        liquid_assets += value
                    elif any(keyword in asset_lower for keyword in fixed_keywords):
                        fixed_assets += value

                data['liquid_assets'] = liquid_assets
                data['fixed_assets'] = fixed_assets

        except Exception as e:
            logger.error(f"Error processing assets/liabilities: {str(e)}")

        return data

    def _process_credit_report(self, file_path: str, text: str) -> Dict[str, Any]:
        """Extract structured data from credit report"""
        data = {
            'credit_score': None,
            'credit_rating': None,
            'total_debt': None,
            'monthly_obligations': None,
            'credit_utilization': None,
            'payment_history': None,
            'accounts': [],
            'inquiries': [],
            'defaults': []
        }

        try:
            # Extract credit score
            score_patterns = [
                r'Credit\s*Score:?\s*(\d{3})',
                r'Score:?\s*(\d{3})',
                r'FICO\s*Score:?\s*(\d{3})'
            ]
            for pattern in score_patterns:
                score_match = re.search(pattern, text, re.IGNORECASE)
                if score_match:
                    data['credit_score'] = int(score_match.group(1))
                    break

            # Extract credit rating
            rating_match = re.search(r'Rating:?\s*(Excellent|Very Good|Good|Fair|Poor)', text, re.IGNORECASE)
            if rating_match:
                data['credit_rating'] = rating_match.group(1)

            # Extract total debt
            debt_patterns = [
                r'Total\s*Debt:?\s*AED\s*([\d,]+\.?\d*)',
                r'Outstanding\s*Balance:?\s*AED\s*([\d,]+\.?\d*)'
            ]
            for pattern in debt_patterns:
                debt_match = re.search(pattern, text, re.IGNORECASE)
                if debt_match:
                    data['total_debt'] = float(debt_match.group(1).replace(',', ''))
                    break

            # Extract monthly obligations
            monthly_match = re.search(r'Monthly\s*Payment:?\s*AED\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
            if monthly_match:
                data['monthly_obligations'] = float(monthly_match.group(1).replace(',', ''))

            # Extract credit utilization
            utilization_match = re.search(r'Credit\s*Utilization:?\s*(\d+)%', text, re.IGNORECASE)
            if utilization_match:
                data['credit_utilization'] = int(utilization_match.group(1))

            # Extract payment history
            payment_match = re.search(r'Payment\s*History:?\s*(\d+)%', text, re.IGNORECASE)
            if payment_match:
                data['payment_history'] = int(payment_match.group(1))

        except Exception as e:
            logger.error(f"Error processing credit report: {str(e)}")

        return data

    def _calculate_confidence(self, extracted_data: Dict[str, Any], document_type: str) -> float:
        """Calculate confidence score based on extracted data completeness"""
        required_fields = {
            'bank_statement': ['account_number', 'account_holder', 'monthly_income'],
            'emirates_id': ['id_number', 'name', 'date_of_birth'],
            'resume': ['name', 'email', 'experience'],
            'assets_liabilities': ['total_assets', 'total_liabilities'],
            'credit_report': ['credit_score', 'total_debt']
        }

        if document_type not in required_fields:
            return 0.5

        required = required_fields[document_type]
        extracted_count = sum(1 for field in required if extracted_data.get(field) is not None)

        base_confidence = extracted_count / len(required)

        # Bonus for additional fields
        total_fields = len(extracted_data)
        non_null_fields = sum(1 for value in extracted_data.values() if value is not None and value != "")
        completeness_bonus = (non_null_fields / total_fields) * 0.2 if total_fields > 0 else 0

        return min(base_confidence + completeness_bonus, 1.0)