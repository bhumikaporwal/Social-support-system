from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage
import logging
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentType, AgentResult
from .document_extraction_agent import DocumentExtractionAgent
from .eligibility_agent import EligibilityAssessmentAgent
from .decision_agent import DecisionRecommendationAgent
from .economic_enablement_agent import EconomicEnablementAgent
from src.services.data_validator import DataValidator

logger = logging.getLogger(__name__)

class ProcessingState(TypedDict):
    """State object for the application processing workflow"""
    application_id: int
    support_type: str
    documents: List[Dict[str, Any]]
    extracted_data: Dict[str, Any]
    validation_result: Dict[str, Any]
    eligibility_assessment: Dict[str, Any]
    final_recommendation: Dict[str, Any]
    economic_enablement: Dict[str, Any]
    processing_errors: List[str]
    current_step: str
    confidence_score: float
    processing_metadata: Dict[str, Any]

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    MANUAL_REVIEW = "manual_review"

@dataclass
class WorkflowResult:
    status: WorkflowStatus
    final_state: ProcessingState
    processing_time: float
    confidence_score: float
    requires_manual_review: bool
    errors: List[str]

class SocialSupportOrchestrator:
    """Master orchestrator for the social support application workflow using LangGraph"""

    def __init__(self, llm_client=None, langfuse_client=None):
        self.llm_client = llm_client
        self.langfuse_client = langfuse_client

        # Initialize agents
        self.document_agent = DocumentExtractionAgent(llm_client, langfuse_client)
        self.eligibility_agent = EligibilityAssessmentAgent(llm_client, langfuse_client)
        self.decision_agent = DecisionRecommendationAgent(llm_client, langfuse_client)
        self.enablement_agent = EconomicEnablementAgent(llm_client, langfuse_client)
        self.validator = DataValidator()

        # Build the workflow graph
        self.workflow_graph = self._build_workflow_graph()
        self.compiled_workflow = self.workflow_graph.compile()

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow for application processing"""

        # Define the workflow graph
        workflow = StateGraph(ProcessingState)

        # Add nodes for each processing step
        workflow.add_node("extract_documents", self._extract_documents)
        workflow.add_node("validate_data", self._validate_data)
        workflow.add_node("assess_eligibility", self._assess_eligibility)
        workflow.add_node("make_decision", self._make_decision)
        workflow.add_node("generate_enablement", self._generate_enablement)
        workflow.add_node("finalize_processing", self._finalize_processing)

        # Set entry point
        workflow.set_entry_point("extract_documents")

        # Define the workflow edges with conditional routing
        workflow.add_conditional_edges(
            "extract_documents",
            self._route_after_extraction,
            {
                "validation": "validate_data",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "validate_data",
            self._route_after_validation,
            {
                "eligibility": "assess_eligibility",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "assess_eligibility",
            self._route_after_eligibility,
            {
                "decision": "make_decision",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "make_decision",
            self._route_after_decision,
            {
                "enablement": "generate_enablement",
                "finalize": "finalize_processing",
                "error": END
            }
        )

        workflow.add_edge("generate_enablement", "finalize_processing")
        workflow.add_edge("finalize_processing", END)

        return workflow

    async def process_application(self, application_data: Dict[str, Any]) -> WorkflowResult:
        """Process a complete social support application through the workflow"""
        import time
        start_time = time.time()

        try:
            # Initialize processing state
            initial_state = ProcessingState(
                application_id=application_data.get('application_id'),
                support_type=application_data.get('support_type', 'both'),
                documents=application_data.get('documents', []),
                extracted_data={},
                validation_result={},
                eligibility_assessment={},
                final_recommendation={},
                economic_enablement={},
                processing_errors=[],
                current_step="initialization",
                confidence_score=0.0,
                processing_metadata={
                    'start_time': start_time,
                    'workflow_version': '1.0'
                }
            )

            # Execute the workflow
            final_state = await self.compiled_workflow.ainvoke(initial_state)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Determine final status
            if final_state.get('processing_errors'):
                status = WorkflowStatus.FAILED
            elif final_state.get('final_recommendation', {}).get('requires_manual_review'):
                status = WorkflowStatus.MANUAL_REVIEW
            else:
                status = WorkflowStatus.COMPLETED

            return WorkflowResult(
                status=status,
                final_state=final_state,
                processing_time=processing_time,
                confidence_score=final_state.get('confidence_score', 0.0),
                requires_manual_review=final_state.get('final_recommendation', {}).get('requires_manual_review', False),
                errors=final_state.get('processing_errors', [])
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            processing_time = time.time() - start_time

            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                final_state=initial_state,
                processing_time=processing_time,
                confidence_score=0.0,
                requires_manual_review=True,
                errors=[f"Workflow execution failed: {str(e)}"]
            )

    async def _extract_documents(self, state: ProcessingState) -> ProcessingState:
        """Document extraction step"""
        logger.info(f"Processing application {state['application_id']}: Document extraction")
        state['current_step'] = "document_extraction"

        try:
            # Prepare input for document extraction agent
            input_data = {
                'documents': state['documents']
            }

            # Run document extraction
            result = await self.document_agent.process(input_data)

            if result.success:
                state['extracted_data'] = result.data.get('extracted_documents', {})
                logger.info(f"Document extraction completed for application {state['application_id']}")
            else:
                state['processing_errors'].extend(result.errors)
                logger.error(f"Document extraction failed for application {state['application_id']}")

        except Exception as e:
            error_msg = f"Document extraction error: {str(e)}"
            state['processing_errors'].append(error_msg)
            logger.error(error_msg)

        return state

    async def _validate_data(self, state: ProcessingState) -> ProcessingState:
        """Data validation step"""
        logger.info(f"Processing application {state['application_id']}: Data validation")
        state['current_step'] = "data_validation"

        try:
            # Run data validation
            validation_result = self.validator.validate_application_data(state['extracted_data'])

            state['validation_result'] = {
                'is_valid': validation_result.is_valid,
                'confidence_score': validation_result.confidence_score,
                'issues': [
                    {
                        'field': issue.field,
                        'issue_type': issue.issue_type,
                        'severity': issue.severity.value,
                        'message': issue.message,
                        'suggestion': issue.suggestion
                    }
                    for issue in validation_result.issues
                ],
                'validated_data': validation_result.validated_data
            }

            logger.info(f"Data validation completed for application {state['application_id']}")

        except Exception as e:
            error_msg = f"Data validation error: {str(e)}"
            state['processing_errors'].append(error_msg)
            logger.error(error_msg)

        return state

    async def _assess_eligibility(self, state: ProcessingState) -> ProcessingState:
        """Eligibility assessment step"""
        logger.info(f"Processing application {state['application_id']}: Eligibility assessment")
        state['current_step'] = "eligibility_assessment"

        try:
            # Prepare input for eligibility assessment
            input_data = {
                'validated_data': state['validation_result'].get('validated_data', {}),
                'support_type': state['support_type']
            }

            # Run eligibility assessment
            result = await self.eligibility_agent.process(input_data)

            if result.success:
                state['eligibility_assessment'] = result.data.get('eligibility_assessment', {})
                logger.info(f"Eligibility assessment completed for application {state['application_id']}")
            else:
                state['processing_errors'].extend(result.errors)
                logger.error(f"Eligibility assessment failed for application {state['application_id']}")

        except Exception as e:
            error_msg = f"Eligibility assessment error: {str(e)}"
            state['processing_errors'].append(error_msg)
            logger.error(error_msg)

        return state

    async def _make_decision(self, state: ProcessingState) -> ProcessingState:
        """Decision making step"""
        logger.info(f"Processing application {state['application_id']}: Decision making")
        state['current_step'] = "decision_making"

        try:
            # Prepare input for decision agent
            input_data = {
                'eligibility_assessment': state['eligibility_assessment'],
                'validation_result': state['validation_result'],
                'extracted_documents': state['extracted_data'],
                'support_type': state['support_type']
            }

            # Run decision making
            result = await self.decision_agent.process(input_data)

            if result.success:
                state['final_recommendation'] = result.data.get('final_recommendation', {})
                state['confidence_score'] = result.confidence
                logger.info(f"Decision making completed for application {state['application_id']}")
            else:
                state['processing_errors'].extend(result.errors)
                logger.error(f"Decision making failed for application {state['application_id']}")

        except Exception as e:
            error_msg = f"Decision making error: {str(e)}"
            state['processing_errors'].append(error_msg)
            logger.error(error_msg)

        return state

    async def _generate_enablement(self, state: ProcessingState) -> ProcessingState:
        """Economic enablement step"""
        logger.info(f"Processing application {state['application_id']}: Economic enablement")
        state['current_step'] = "economic_enablement"

        try:
            # Prepare input for economic enablement agent
            input_data = {
                'validated_data': state['validation_result'].get('validated_data', {}),
                'eligibility_assessment': state['eligibility_assessment'],
                'final_recommendation': state['final_recommendation']
            }

            # Run economic enablement recommendations
            result = await self.enablement_agent.process(input_data)

            if result.success:
                state['economic_enablement'] = result.data
                logger.info(f"Economic enablement completed for application {state['application_id']}")
            else:
                state['processing_errors'].extend(result.errors)
                logger.error(f"Economic enablement failed for application {state['application_id']}")

        except Exception as e:
            error_msg = f"Economic enablement error: {str(e)}"
            state['processing_errors'].append(error_msg)
            logger.error(error_msg)

        return state

    async def _finalize_processing(self, state: ProcessingState) -> ProcessingState:
        """Finalize processing step"""
        logger.info(f"Processing application {state['application_id']}: Finalizing")
        state['current_step'] = "completed"

        # Calculate final confidence score
        confidence_factors = [
            state['validation_result'].get('confidence_score', 0.5),
            state['confidence_score'],
            1.0 if not state['processing_errors'] else 0.5
        ]
        state['confidence_score'] = sum(confidence_factors) / len(confidence_factors)

        # Add completion metadata
        state['processing_metadata']['completion_time'] = logger.info.__module__  # Replace with actual timestamp
        state['processing_metadata']['total_steps'] = 6
        state['processing_metadata']['error_count'] = len(state['processing_errors'])

        logger.info(f"Application {state['application_id']} processing completed with confidence {state['confidence_score']:.2f}")

        return state

    # Routing functions for conditional edges

    def _route_after_extraction(self, state: ProcessingState) -> str:
        """Route after document extraction"""
        if state['processing_errors'] or not state['extracted_data']:
            return "error"
        return "validation"

    def _route_after_validation(self, state: ProcessingState) -> str:
        """Route after data validation"""
        if state['processing_errors']:
            return "error"

        # Check for critical validation errors
        critical_errors = [
            issue for issue in state['validation_result'].get('issues', [])
            if issue.get('severity') == 'error'
        ]

        if critical_errors:
            state['processing_errors'].append("Critical validation errors found")
            return "error"

        return "eligibility"

    def _route_after_eligibility(self, state: ProcessingState) -> str:
        """Route after eligibility assessment"""
        if state['processing_errors'] or not state['eligibility_assessment']:
            return "error"
        return "decision"

    def _route_after_decision(self, state: ProcessingState) -> str:
        """Route after decision making"""
        if state['processing_errors']:
            return "error"

        # Check if economic enablement is needed
        decision = state['final_recommendation'].get('decision', 'decline')
        support_type = state['support_type']

        if decision in ['approve', 'conditional_approve'] and support_type in ['economic_enablement', 'both']:
            return "enablement"

        return "finalize"

    async def get_workflow_status(self, application_id: int) -> Dict[str, Any]:
        """Get current workflow status for an application"""
        # This would typically check a database or cache
        # For demo purposes, return a sample status
        return {
            'application_id': application_id,
            'current_step': 'processing',
            'progress_percentage': 50,
            'estimated_completion': '2 minutes',
            'last_updated': '2024-01-01T00:00:00Z'
        }

    def get_workflow_diagram(self) -> str:
        """Return a text representation of the workflow"""
        return """
        Social Support Application Workflow:

        1. Document Extraction
           ↓
        2. Data Validation
           ↓
        3. Eligibility Assessment
           ↓
        4. Decision Making
           ↓ (if approved for economic enablement)
        5. Economic Enablement Recommendations
           ↓
        6. Finalize Processing

        Error conditions lead to immediate termination and manual review.
        """