import streamlit as st
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from services.llm_client import SocialSupportChatbot, OllamaClient
from agents.orchestrator import SocialSupportOrchestrator
from models import ApplicationStatus, SupportType

# Page configuration
st.set_page_config(
    page_title="UAE Social Support System",
    page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e6da4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .status-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "application_data" not in st.session_state:
    st.session_state.application_data = {}
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

def initialize_services():
    """Initialize AI services"""
    if st.session_state.chatbot is None:
        llm_client = OllamaClient()
        st.session_state.chatbot = SocialSupportChatbot(llm_client)
        st.session_state.orchestrator = SocialSupportOrchestrator(llm_client)

def main():
    """Main application function"""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇦🇪 UAE Social Support System</h1>
        <p>AI-Powered Application Processing & Support</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize services
    initialize_services()

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        # Initialize page selection if not exists
        if "page_selection" not in st.session_state:
            st.session_state.page_selection = "🏠 Home"

        page = st.selectbox(
            "Select Page",
            ["🏠 Home", "📝 New Application", "💬 AI Assistant", "📊 Dashboard", "📋 Application Status", "ℹ️ Help"],
            index=["🏠 Home", "📝 New Application", "💬 AI Assistant", "📊 Dashboard", "📋 Application Status", "ℹ️ Help"].index(st.session_state.page_selection)
        )

        # Update session state when selectbox changes
        st.session_state.page_selection = page

    # Route to different pages
    if page == "🏠 Home":
        home_page()
    elif page == "📝 New Application":
        new_application_page()
    elif page == "💬 AI Assistant":
        chat_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "📋 Application Status":
        status_page()
    elif page == "ℹ️ Help":
        help_page()

def home_page():
    """Home page with overview"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Welcome to the UAE Social Support System")
        st.write("""
        Our AI-powered system helps UAE residents access social support services quickly and efficiently.
        The automated processing system can handle up to 99% of applications within minutes.
        """)

        st.subheader("How it works:")
        st.write("""
        1. **Submit Application**: Upload your documents and fill out the form
        2. **AI Processing**: Our AI agents extract and validate your information
        3. **Eligibility Assessment**: Automated evaluation against UAE criteria
        4. **Decision**: Receive approval or recommendations within minutes
        5. **Support Programs**: Get matched with suitable training and job opportunities
        """)

        if st.button("Start New Application", type="primary", use_container_width=True):
            st.session_state.page_selection = "📝 New Application"
            st.rerun()

    with col2:
        st.subheader("Quick Stats")

        # Mock statistics
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Applications Today", "147", "+12")
            st.metric("Approval Rate", "78%", "+5%")
        with col2_2:
            st.metric("Avg Processing Time", "3.2 min", "-45s")
            st.metric("Satisfaction Score", "4.6/5", "+0.2")

def new_application_page():
    """New application submission page"""
    st.header("Submit New Application")

    # Progress indicator
    if "application_step" not in st.session_state:
        st.session_state.application_step = 1

    progress = st.progress((st.session_state.application_step - 1) / 4)
    st.write(f"Step {st.session_state.application_step} of 4")

    if st.session_state.application_step == 1:
        personal_info_step()
    elif st.session_state.application_step == 2:
        document_upload_step()
    elif st.session_state.application_step == 3:
        review_step()
    elif st.session_state.application_step == 4:
        processing_step()

def personal_info_step():
    """Personal information collection step"""
    st.subheader("Personal Information")

    with st.form("personal_info_form"):
        col1, col2 = st.columns(2)

        with col1:
            emirates_id = st.text_input("Emirates ID", placeholder="784-1988-1234567-8")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            date_of_birth = st.date_input("Date of Birth")
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            nationality = st.selectbox("Nationality", ["UAE", "Other GCC", "Arab", "Asian", "African", "European", "American", "Other"])
            email = st.text_input("Email Address")
            phone = st.text_input("Phone Number", placeholder="+971 50 123 4567")
            emirate = st.selectbox("Emirate", ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

        # Support type selection
        st.subheader("Support Type Requested")
        support_type = st.radio(
            "What type of support are you seeking?",
            ["Financial Support", "Economic Enablement", "Both"],
            help="Financial Support: Monthly financial assistance. Economic Enablement: Training and job placement support."
        )

        # Family information
        st.subheader("Family Information")
        col3, col4 = st.columns(2)
        with col3:
            family_size = st.number_input("Family Size", min_value=1, max_value=20, value=1)
        with col4:
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

        submitted = st.form_submit_button("Next Step")

        if submitted:
            # Validate required fields
            if not all([emirates_id, first_name, last_name, email, phone]):
                st.error("Please fill in all required fields.")
                return

            # Store data
            st.session_state.application_data.update({
                "emirates_id": emirates_id,
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": date_of_birth.isoformat(),
                "gender": gender,
                "nationality": nationality,
                "email": email,
                "phone": phone,
                "emirate": emirate,
                "marital_status": marital_status,
                "support_type": support_type.lower().replace(" ", "_"),
                "family_size": family_size,
                "dependents": dependents
            })

            st.session_state.application_step = 2
            st.rerun()

def document_upload_step():
    """Document upload step"""
    st.subheader("Document Upload")

    st.write("Please upload the following documents for processing:")

    # Document requirements
    required_docs = {
        "Emirates ID": "Clear photo or scan of both sides of your Emirates ID",
        "Bank Statement": "Latest 3 months bank statement (PDF format preferred)",
        "Resume/CV": "Current resume or CV",
        "Assets & Liabilities": "Excel file listing your assets and liabilities",
        "Credit Report": "Latest credit report (if available)"
    }

    uploaded_files = {}

    for doc_type, description in required_docs.items():
        st.write(f"**{doc_type}**")
        st.write(f"*{description}*")

        file = st.file_uploader(
            f"Upload {doc_type}",
            type=["pdf", "jpg", "jpeg", "png", "xlsx", "xls", "docx", "doc"],
            key=f"upload_{doc_type.lower().replace(' ', '_')}"
        )

        if file is not None:
            uploaded_files[doc_type.lower().replace(' & ', '_').replace(' ', '_')] = {
                "file_name": file.name,
                "file_size": file.size,
                "file_type": file.type,
                "file_content": file.getvalue()
            }
            st.success(f"✅ {doc_type} uploaded successfully")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step"):
            st.session_state.application_step = 1
            st.rerun()

    with col2:
        if st.button("Next Step", type="primary"):
            if len(uploaded_files) < 2:  # At least Emirates ID and Bank Statement
                st.error("Please upload at least Emirates ID and Bank Statement to proceed.")
                return

            st.session_state.application_data["uploaded_files"] = uploaded_files
            st.session_state.application_step = 3
            st.rerun()

def review_step():
    """Review application before submission"""
    st.subheader("Review Application")

    st.write("Please review your information before submission:")

    # Personal information review
    with st.expander("Personal Information", expanded=True):
        data = st.session_state.application_data
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Name:** {data['first_name']} {data['last_name']}")
            st.write(f"**Emirates ID:** {data['emirates_id']}")
            st.write(f"**Email:** {data['email']}")
            st.write(f"**Phone:** {data['phone']}")

        with col2:
            st.write(f"**Nationality:** {data['nationality']}")
            st.write(f"**Emirate:** {data['emirate']}")
            st.write(f"**Support Type:** {data['support_type']}")
            st.write(f"**Family Size:** {data['family_size']}")

    # Documents review
    with st.expander("Uploaded Documents", expanded=True):
        uploaded_files = st.session_state.application_data.get("uploaded_files", {})
        for doc_type, file_info in uploaded_files.items():
            st.write(f"✅ **{doc_type.replace('_', ' ').title()}**: {file_info['file_name']}")

    # Terms and conditions
    st.subheader("Terms and Conditions")
    terms_accepted = st.checkbox(
        "I agree to the terms and conditions and confirm that all information provided is accurate.",
        key="terms_checkbox"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous Step"):
            st.session_state.application_step = 2
            st.rerun()

    with col2:
        if st.button("Submit Application", type="primary", disabled=not terms_accepted):
            if not terms_accepted:
                st.error("Please accept the terms and conditions.")
                return

            st.session_state.application_step = 4
            st.rerun()

def processing_step():
    """Application processing step"""
    st.subheader("Processing Application")

    # Simulate processing with progress
    if "processing_started" not in st.session_state:
        st.session_state.processing_started = False

    if not st.session_state.processing_started:
        st.session_state.processing_started = True
        st.rerun()

    # Processing steps
    steps = [
        "Validating documents...",
        "Extracting information...",
        "Assessing eligibility...",
        "Making decision...",
        "Generating recommendations..."
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(1)  # Simulate processing time

    # Show results
    st.success("✅ Application processed successfully!")

    # Mock results
    st.subheader("Application Result")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="success-message">
            <strong>Application Approved!</strong><br>
            Your application for social support has been approved based on our automated assessment.
        </div>
        """, unsafe_allow_html=True)

        st.write("**Support Details:**")
        st.write("- Monthly Financial Support: 3,500 AED")
        st.write("- Duration: 6 months")
        st.write("- Economic Enablement Budget: 8,000 AED")
        st.write("- Recommended Training: Digital Skills Bootcamp")

    with col2:
        st.metric("Confidence Score", "87%")
        st.metric("Processing Time", "2.3 minutes")
        st.metric("Application ID", "APP-2024-001")

    if st.button("Start Over", type="primary"):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key.startswith("application"):
                del st.session_state[key]
        st.session_state.application_step = 1
        st.rerun()

def chat_page():
    """AI Assistant chat page"""
    st.header("💬 AI Assistant")
    st.write("Ask me anything about the social support application process!")

    # Chat container
    chat_container = st.container()

    # Display conversation
    with chat_container:
        for message in st.session_state.conversation:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.conversation.append({"role": "user", "content": user_input})

        # Get AI response (mock for demo)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Mock AI response
                if "eligibility" in user_input.lower():
                    response = """For financial support eligibility in the UAE, you must meet these criteria:

- Be a UAE resident with valid Emirates ID
- Monthly income below 15,000 AED
- Net worth less than 500,000 AED
- Age between 18-65 years
- Debt-to-income ratio below 60%

For economic enablement support, the income threshold is higher (25,000 AED) and focuses on skills development and job placement."""

                elif "documents" in user_input.lower():
                    response = """You'll need to upload these documents:

**Required:**
- Emirates ID (both sides)
- Bank statements (last 3 months)
- Resume/CV

**Optional but helpful:**
- Assets and liabilities statement
- Credit report
- Educational certificates

All documents should be clear and in PDF, image, or Excel format."""

                else:
                    response = """I'm here to help you with your social support application! I can provide information about:

- Eligibility criteria
- Required documents
- Application process
- Processing timelines
- Support programs available
- Troubleshooting common issues

What would you like to know more about?"""

                st.write(response)

        # Add AI response to conversation
        st.session_state.conversation.append({"role": "assistant", "content": response})
        st.rerun()

def dashboard_page():
    """Analytics dashboard page"""
    st.header("📊 System Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Applications", "1,247", "+23 today")
    with col2:
        st.metric("Approval Rate", "78%", "+5%")
    with col3:
        st.metric("Avg Processing Time", "3.2 min", "-45s")
    with col4:
        st.metric("User Satisfaction", "4.6/5", "+0.2")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Applications by Status")
        # Mock data
        status_data = pd.DataFrame({
            'Status': ['Approved', 'Pending', 'Declined', 'Under Review'],
            'Count': [456, 123, 78, 45]
        })
        fig = px.pie(status_data, values='Count', names='Status',
                    title="Application Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Processing Time Trend")
        # Mock data
        time_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Avg Time (min)': [4.2, 3.8, 3.5, 3.2, 3.0, 2.8, 3.1, 3.4, 3.2, 2.9] * 3
        })
        fig = px.line(time_data, x='Date', y='Avg Time (min)',
                     title="Average Processing Time")
        st.plotly_chart(fig, use_container_width=True)

def status_page():
    """Application status check page"""
    st.header("📋 Check Application Status")

    # Status lookup
    col1, col2 = st.columns([2, 1])

    with col1:
        application_id = st.text_input("Enter Application ID", placeholder="APP-2024-001")
        emirates_id = st.text_input("Enter Emirates ID", placeholder="784-1988-1234567-8")

        if st.button("Check Status", type="primary"):
            if application_id and emirates_id:
                # Mock status
                st.markdown("""
                <div class="status-card">
                    <h3>Application Status: ✅ Approved</h3>
                    <p><strong>Application ID:</strong> APP-2024-001</p>
                    <p><strong>Submitted:</strong> January 15, 2024</p>
                    <p><strong>Processed:</strong> January 15, 2024 (2.3 minutes)</p>
                    <p><strong>Decision:</strong> Approved for Financial Support</p>
                    <p><strong>Monthly Amount:</strong> 3,500 AED</p>
                    <p><strong>Duration:</strong> 6 months</p>
                    <p><strong>Next Payment:</strong> February 1, 2024</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Please enter both Application ID and Emirates ID.")

    with col2:
        st.info("""
        **Need Help?**

        Contact our support team:
        - Phone: 800-UAE-HELP
        - Email: support@uae.gov.ae
        - Hours: 8 AM - 8 PM
        """)

def help_page():
    """Help and FAQ page"""
    st.header("ℹ️ Help & FAQ")

    # FAQ sections
    with st.expander("General Information", expanded=True):
        st.write("""
        **Q: What is the UAE Social Support System?**
        A: An AI-powered platform that automates the processing of social support applications for UAE residents.

        **Q: How long does processing take?**
        A: Most applications are processed within 3-5 minutes using our automated system.

        **Q: Is my data secure?**
        A: Yes, all data is encrypted and processed locally in compliance with UAE data protection laws.
        """)

    with st.expander("Eligibility Requirements"):
        st.write("""
        **Financial Support Eligibility:**
        - UAE resident with valid Emirates ID
        - Monthly income below 15,000 AED
        - Net worth less than 500,000 AED
        - Age between 18-65 years

        **Economic Enablement Eligibility:**
        - UAE resident with valid Emirates ID
        - Monthly income below 25,000 AED
        - Age between 18-55 years
        - Interest in skills development
        """)

    with st.expander("Required Documents"):
        st.write("""
        **Mandatory Documents:**
        - Emirates ID (both sides)
        - Bank statements (last 3 months)
        - Resume or CV

        **Optional Documents:**
        - Assets and liabilities statement
        - Credit report
        - Educational certificates
        - Employment contracts
        """)

    with st.expander("Technical Support"):
        st.write("""
        **Common Issues:**
        - Document upload problems: Ensure files are under 10MB
        - Processing delays: Check internet connection
        - Login issues: Clear browser cache

        **Contact Support:**
        - Technical Help: tech-support@uae.gov.ae
        - General Inquiries: info@uae.gov.ae
        - Phone: 800-UAE-HELP
        """)

if __name__ == "__main__":
    main()