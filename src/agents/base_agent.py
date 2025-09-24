from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from langchain.schema import BaseMessage
from langfuse import Langfuse

logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    DOCUMENT_EXTRACTION = "document_extraction"
    DATA_VALIDATION = "data_validation"
    ELIGIBILITY_ASSESSMENT = "eligibility_assessment"
    DECISION_RECOMMENDATION = "decision_recommendation"
    ECONOMIC_ENABLEMENT = "economic_enablement"
    MASTER_ORCHESTRATOR = "master_orchestrator"

@dataclass
class AgentResult:
    success: bool
    data: Dict[str, Any]
    reasoning: str
    confidence: float
    errors: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """Base class for all AI agents in the system"""

    def __init__(self, agent_type: AgentType, llm_client=None, langfuse_client: Optional[Langfuse] = None):
        self.agent_type = agent_type
        self.llm_client = llm_client
        self.langfuse_client = langfuse_client
        self.logger = logging.getLogger(f"{__name__}.{agent_type.value}")

    @abstractmethod
    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Main processing method that each agent must implement"""
        pass

    def _create_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """Create a prompt from template and variables"""
        try:
            return template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing variable in prompt template: {e}")
            raise

    async def _call_llm(self, prompt: str, system_message: str = None) -> str:
        """Call the LLM with proper error handling and observability"""
        try:
            # Create trace if Langfuse is available
            trace = None
            if self.langfuse_client:
                trace = self.langfuse_client.trace(
                    name=f"{self.agent_type.value}_llm_call",
                    input={"prompt": prompt, "system": system_message}
                )

            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # Call LLM (assuming OpenAI-compatible interface)
            response = await self.llm_client.chat.completions.create(
                model="llama2:7b-chat",  # Will be configured via Ollama
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )

            result = response.choices[0].message.content

            # Log to Langfuse if available
            if trace:
                trace.update(output=result)

            return result

        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise

    def _calculate_confidence(self, result_data: Dict[str, Any], validation_checks: List[bool]) -> float:
        """Calculate confidence score based on data quality and validation results"""
        # Base confidence from validation checks
        validation_score = sum(validation_checks) / len(validation_checks) if validation_checks else 0.5

        # Data completeness score
        total_fields = len(result_data)
        filled_fields = sum(1 for value in result_data.values() if value is not None and value != "")
        completeness_score = filled_fields / total_fields if total_fields > 0 else 0

        # Weighted average
        confidence = (validation_score * 0.7) + (completeness_score * 0.3)
        return min(max(confidence, 0.0), 1.0)

    def log_decision(self, decision: str, reasoning: str, confidence: float):
        """Log agent decision for audit trail"""
        log_entry = {
            "agent_type": self.agent_type.value,
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
            "timestamp": logger.info.__module__  # This should be replaced with actual timestamp
        }
        self.logger.info(f"Agent Decision: {log_entry}")