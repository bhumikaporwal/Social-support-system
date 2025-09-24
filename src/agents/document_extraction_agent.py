from typing import Dict, Any, Optional, List
import asyncio
from .base_agent import BaseAgent, AgentType, AgentResult
from src.services.document_processor import DocumentProcessor

class DocumentExtractionAgent(BaseAgent):
    """Agent responsible for extracting structured data from documents"""

    def __init__(self, llm_client=None, langfuse_client=None):
        super().__init__(AgentType.DOCUMENT_EXTRACTION, llm_client, langfuse_client)
        self.document_processor = DocumentProcessor()

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Extract data from uploaded documents"""
        try:
            documents = input_data.get('documents', [])
            extracted_data = {}
            all_errors = []
            confidence_scores = []

            for doc in documents:
                file_path = doc.get('file_path')
                document_type = doc.get('document_type')

                if not file_path or not document_type:
                    error = f"Missing file_path or document_type for document: {doc}"
                    all_errors.append(error)
                    continue

                # Process document
                extraction_result = self.document_processor.process_document(file_path, document_type)

                if extraction_result.processing_status == "completed":
                    extracted_data[document_type] = extraction_result.extracted_data
                    confidence_scores.append(extraction_result.confidence_score)

                    # Use LLM to enhance extraction if confidence is low
                    if extraction_result.confidence_score < 0.7:
                        enhanced_data = await self._enhance_extraction_with_llm(
                            extraction_result.extracted_text,
                            extraction_result.extracted_data,
                            document_type
                        )
                        extracted_data[document_type].update(enhanced_data)

                else:
                    all_errors.extend(extraction_result.errors)

            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            # Generate reasoning
            reasoning = self._generate_extraction_reasoning(extracted_data, confidence_scores, all_errors)

            return AgentResult(
                success=len(extracted_data) > 0,
                data={"extracted_documents": extracted_data},
                reasoning=reasoning,
                confidence=overall_confidence,
                errors=all_errors,
                metadata={"processed_count": len(documents), "successful_count": len(extracted_data)}
            )

        except Exception as e:
            self.logger.error(f"Document extraction failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                reasoning=f"Document extraction failed due to: {str(e)}",
                confidence=0.0,
                errors=[str(e)]
            )

    async def _enhance_extraction_with_llm(self, text: str, extracted_data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Use LLM to enhance low-confidence extractions"""
        try:
            prompt_template = """
            Analyze the following {document_type} document text and extract missing or improve existing information.

            Current extracted data:
            {extracted_data}

            Document text:
            {text}

            Please provide additional or corrected information in JSON format that matches the structure of current extracted data.
            Focus on filling missing fields and correcting obvious errors.
            """

            prompt = self._create_prompt(prompt_template, {
                "document_type": document_type,
                "extracted_data": str(extracted_data),
                "text": text[:2000]  # Limit text length for LLM
            })

            system_message = f"You are an expert at extracting structured information from {document_type} documents. Provide accurate, complete information in JSON format."

            response = await self._call_llm(prompt, system_message)

            # Parse LLM response (simplified - in production, use proper JSON parsing)
            try:
                import json
                enhanced_data = json.loads(response)
                return enhanced_data
            except json.JSONDecodeError:
                self.logger.warning("LLM response was not valid JSON")
                return {}

        except Exception as e:
            self.logger.error(f"LLM enhancement failed: {str(e)}")
            return {}

    def _generate_extraction_reasoning(self, extracted_data: Dict[str, Any], confidence_scores: List[float], errors: List[str]) -> str:
        """Generate reasoning for extraction process"""
        reasoning_parts = []

        if extracted_data:
            reasoning_parts.append(f"Successfully extracted data from {len(extracted_data)} document types.")

            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            if avg_confidence > 0.8:
                reasoning_parts.append("High confidence extraction achieved across all documents.")
            elif avg_confidence > 0.6:
                reasoning_parts.append("Moderate confidence extraction. Some fields may need verification.")
            else:
                reasoning_parts.append("Low confidence extraction. Manual review recommended.")

            # Document-specific insights
            for doc_type, data in extracted_data.items():
                filled_fields = sum(1 for v in data.values() if v is not None and v != "")
                total_fields = len(data)
                reasoning_parts.append(f"{doc_type}: {filled_fields}/{total_fields} fields extracted.")

        if errors:
            reasoning_parts.append(f"Encountered {len(errors)} errors during processing.")

        return " ".join(reasoning_parts)