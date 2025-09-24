import httpx
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from src.config import settings

logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    LLAMA2_7B = "llama2:7b-chat"
    LLAMA2_13B = "llama2:13b-chat"
    MISTRAL_7B = "mistral:7b"
    CODELLAMA = "codellama:7b"

@dataclass
class ChatMessage:
    role: str  # system, user, assistant
    content: str

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    response_time: float

class OllamaClient:
    """Client for interacting with locally hosted Ollama LLM"""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.default_model = model or settings.ollama_model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> LLMResponse:
        """Generate chat completion using Ollama"""
        import time
        start_time = time.time()

        try:
            model_name = model or self.default_model

            # Format messages for Ollama API
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            payload = {
                "model": model_name,
                "messages": formatted_messages,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()

            if stream:
                return self._handle_streaming_response(response, model_name, start_time)
            else:
                result = response.json()
                response_time = time.time() - start_time

                return LLMResponse(
                    content=result["message"]["content"],
                    model=model_name,
                    tokens_used=result.get("eval_count", 0),
                    response_time=response_time
                )

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    async def generate_completion(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate text completion using Ollama"""
        import time
        start_time = time.time()

        try:
            model_name = model or self.default_model

            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            response_time = time.time() - start_time

            return LLMResponse(
                content=result["response"],
                model=model_name,
                tokens_used=result.get("eval_count", 0),
                response_time=response_time
            )

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion responses"""
        try:
            model_name = model or self.default_model

            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            payload = {
                "model": model_name,
                "messages": formatted_messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                yield chunk["message"]["content"]
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Error streaming chat completion: {e}")
            raise

    async def check_model_availability(self, model: str = None) -> bool:
        """Check if a model is available in Ollama"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            models = response.json()
            model_names = [model["name"] for model in models.get("models", [])]

            target_model = model or self.default_model
            return target_model in model_names

        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    async def pull_model(self, model: str) -> bool:
        """Pull/download a model in Ollama"""
        try:
            payload = {"name": model}

            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json=payload
            )
            response.raise_for_status()

            logger.info(f"Model {model} pulled successfully")
            return True

        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

class SocialSupportChatbot:
    """Specialized chatbot for social support applications"""

    def __init__(self, llm_client: OllamaClient = None):
        self.llm_client = llm_client or OllamaClient()
        self.conversation_history: List[ChatMessage] = []
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the social support chatbot"""
        return """You are a helpful AI assistant for UAE Social Support applications.

Your role: Guide users through applications, explain eligibility, help with documents, and provide clear information about the process.

Be professional, empathetic, and helpful. If unsure, direct to human support."""

    async def chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a chat message and return response"""
        try:
            # Add randomness to the user message processing
            import random

            # Add user message to history
            self.conversation_history.append(ChatMessage(role="user", content=user_message))

            # Prepare messages for LLM with slight variations
            system_variations = [
                self.system_prompt,
                f"{self.system_prompt}\n\nFocus on being particularly helpful and detailed in your response.",
                f"{self.system_prompt}\n\nProvide practical examples when possible.",
                f"{self.system_prompt}\n\nBe conversational and engaging in your response."
            ]

            messages = [ChatMessage(role="system", content=random.choice(system_variations))]

            # Add context if provided
            if context:
                context_message = self._format_context(context)
                messages.append(ChatMessage(role="system", content=context_message))

            # Add conversation history (keep last 10 exchanges to manage context length)
            messages.extend(self.conversation_history[-20:])

            # Vary temperature slightly for each request
            temp_variation = random.uniform(0.7, 0.9)

            # Get LLM response
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=temp_variation,  # Variable temperature for more varied responses
                max_tokens=1500
            )

            # Add assistant response to history
            self.conversation_history.append(
                ChatMessage(role="assistant", content=response.content)
            )

            return response.content

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team for assistance."

    async def stream_chat(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response for real-time interaction"""
        try:
            import random

            # Add user message to history
            self.conversation_history.append(ChatMessage(role="user", content=user_message))

            # Prepare messages with variations
            system_variations = [
                self.system_prompt,
                f"{self.system_prompt}\n\nFocus on being particularly helpful and detailed in your response.",
                f"{self.system_prompt}\n\nProvide practical examples when possible.",
                f"{self.system_prompt}\n\nBe conversational and engaging in your response."
            ]

            messages = [ChatMessage(role="system", content=random.choice(system_variations))]

            if context:
                context_message = self._format_context(context)
                messages.append(ChatMessage(role="system", content=context_message))

            messages.extend(self.conversation_history[-20:])

            # Variable temperature
            temp_variation = random.uniform(0.7, 0.9)

            # Stream response
            full_response = ""
            async for chunk in self.llm_client.stream_chat_completion(
                messages=messages,
                temperature=temp_variation,
                max_tokens=1500
            ):
                full_response += chunk
                yield chunk

            # Add complete response to history
            self.conversation_history.append(
                ChatMessage(role="assistant", content=full_response)
            )

        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            yield "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team for assistance."

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the LLM"""
        context_parts = []

        if 'application_status' in context:
            context_parts.append(f"Application Status: {context['application_status']}")

        if 'current_step' in context:
            context_parts.append(f"Current Processing Step: {context['current_step']}")

        if 'documents_uploaded' in context:
            docs = context['documents_uploaded']
            context_parts.append(f"Documents Uploaded: {', '.join(docs) if docs else 'None'}")

        if 'missing_documents' in context:
            missing = context['missing_documents']
            if missing:
                context_parts.append(f"Missing Documents: {', '.join(missing)}")

        if 'eligibility_result' in context:
            result = context['eligibility_result']
            context_parts.append(f"Eligibility Assessment: {result}")

        if 'decision' in context:
            decision = context['decision']
            context_parts.append(f"Application Decision: {decision}")

        return "\n".join(context_parts) if context_parts else ""

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            'message_count': len(self.conversation_history),
            'last_user_message': next(
                (msg.content for msg in reversed(self.conversation_history) if msg.role == "user"),
                None
            ),
            'last_assistant_message': next(
                (msg.content for msg in reversed(self.conversation_history) if msg.role == "assistant"),
                None
            )
        }

    async def get_application_guidance(self, support_type: str = "both") -> str:
        """Get specific guidance for application process"""
        guidance_prompt = f"""
        Provide detailed guidance for applying for {support_type} social support. Include:
        1. Required documents
        2. Eligibility criteria
        3. Application process steps
        4. What to expect during assessment
        5. Timeline for decision

        Make it clear and easy to understand for UAE residents.
        """

        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=guidance_prompt)
        ]

        response = await self.llm_client.chat_completion(messages=messages, temperature=0.7)
        return response.content