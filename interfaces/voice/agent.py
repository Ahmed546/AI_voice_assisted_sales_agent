"""
SalesGPT Core Agent Module

This module contains the main SalesGPT Agent class that coordinates all components
and manages the conversation flow of the AI sales agent.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, AsyncGenerator

from pydantic import BaseModel, Field

from salesgpt.core.conversation import ConversationContext, ConversationTurn, SalesStage
from salesgpt.core.memory import MemoryManager
from salesgpt.llm.provider import LLMProvider
from salesgpt.knowledge.base import KnowledgeBase
from salesgpt.actions.base import ActionEngine
from salesgpt.exceptions import SalesGPTError
from salesgpt.config import SalesGPTConfig
from salesgpt.utils.metrics import track_metric

logger = logging.getLogger(__name__)

class SalesGPTAgent:
    """
    The main SalesGPT Agent class that coordinates all components
    and manages the AI sales agent's conversation flow.
    """
    
    def __init__(self, config: SalesGPTConfig):
        """
        Initialize the SalesGPT Agent with the necessary components.

        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        
        # Initialize components
        self.llm_provider: LLMProvider = self._init_llm_provider()
        self.knowledge_base: KnowledgeBase = self._init_knowledge_base()
        self.action_engine: ActionEngine = self._init_action_engine()
        self.memory_manager: MemoryManager = self._init_memory_manager()
        
        # Conversation storage
        self.conversations: Dict[str, ConversationContext] = {}
        
        logger.info("SalesGPT Agent initialized with: LLM=%s, Knowledge Base=%s", 
                   self.config.llm.provider, self.config.knowledge_base.type)

    def _init_llm_provider(self) -> LLMProvider:
        """Initialize the appropriate LLM provider based on configuration"""
        from salesgpt.llm.openai import OpenAIProvider
        from salesgpt.llm.anthropic import AnthropicProvider
        from salesgpt.llm.huggingface import HuggingFaceProvider
        from salesgpt.llm.local import LocalModelProvider
        
        provider_name = self.config.llm.provider.lower()
        
        if provider_name == "openai":
            return OpenAIProvider(self.config.llm)
        elif provider_name == "anthropic":
            return AnthropicProvider(self.config.llm)
        elif provider_name == "huggingface":
            return HuggingFaceProvider(self.config.llm)
        elif provider_name == "local":
            return LocalModelProvider(self.config.llm)
        else:
            raise SalesGPTError(f"Unsupported LLM provider: {provider_name}")

    def _init_knowledge_base(self) -> KnowledgeBase:
        """Initialize the knowledge base"""
        from salesgpt.knowledge.document_store import DocumentStore
        from salesgpt.knowledge.vector_store import VectorStore
        
        kb_type = self.config.knowledge_base.type.lower()
        
        if kb_type == "document":
            return DocumentStore(self.config.knowledge_base)
        elif kb_type == "vector":
            return VectorStore(self.config.knowledge_base)
        else:
            raise SalesGPTError(f"Unsupported knowledge base type: {kb_type}")

    def _init_action_engine(self) -> ActionEngine:
        """Initialize the action engine"""
        return ActionEngine(self.config.actions)

    def _init_memory_manager(self) -> MemoryManager:
        """Initialize the memory manager"""
        return MemoryManager(self.config.memory)

    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Create a new conversation instance.
        
        Args:
            conversation_id: Optional ID for the conversation. If not provided, a new UUID will be generated.
            
        Returns:
            The conversation ID.
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            
        # Create new conversation context
        self.conversations[conversation_id] = ConversationContext(
            conversation_id=conversation_id,
            created_at=datetime.now(),
            current_stage=SalesStage.INTRODUCTION
        )
        
        logger.info("Created new conversation with ID: %s", conversation_id)
        track_metric("conversation_created", {"conversation_id": conversation_id})
        
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve.
            
        Returns:
            The conversation context if found, None otherwise.
        """
        return self.conversations.get(conversation_id)
    
    async def process_message(self, conversation_id: str, message: str) -> str:
        """
        Process a customer message and generate a response.
        
        Args:
            conversation_id: The ID of the conversation.
            message: The customer's message text.
            
        Returns:
            The agent's response text.
        """
        start_time = datetime.now()
        
        # Get or create conversation
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            conversation_id = self.create_conversation(conversation_id)
            conversation = self.get_conversation(conversation_id)
        
        # Add customer message to history
        customer_turn = ConversationTurn(
            speaker="customer",
            text=message,
            timestamp=datetime.now()
        )
        conversation.conversation_history.append(customer_turn)
        
        try:
            # Prepare context for LLM
            context = await self._prepare_context(conversation)
            
            # Generate response
            response = await self.llm_provider.generate_response(message, context)
            
            # Update conversation with agent response
            agent_turn = ConversationTurn(
                speaker="agent",
                text=response,
                timestamp=datetime.now()
            )
            conversation.conversation_history.append(agent_turn)
            
            # Process response for actions and context updates
            await self._process_response(conversation, response)
            
            # Update stage and extract information
            await self._update_conversation_context(conversation)
            
            # Track metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            track_metric("message_processed", {
                "conversation_id": conversation_id,
                "processing_time": processing_time,
                "current_stage": conversation.current_stage.value,
                "message_length": len(message),
                "response_length": len(response)
            })
            
            return response
            
        except Exception as e:
            logger.exception("Error processing message: %s", str(e))
            track_metric("message_error", {
                "conversation_id": conversation_id,
                "error": str(e)
            })
            raise SalesGPTError(f"Failed to process message: {str(e)}") from e
    
    async def process_message_stream(self, conversation_id: str, message: str) -> AsyncGenerator[str, None]:
        """
        Process a customer message and generate a streaming response.
        
        Args:
            conversation_id: The ID of the conversation.
            message: The customer's message text.
            
        Yields:
            Chunks of the agent's response as they are generated.
        """
        start_time = datetime.now()
        
        # Get or create conversation
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            conversation_id = self.create_conversation(conversation_id)
            conversation = self.get_conversation(conversation_id)
        
        # Add customer message to history
        customer_turn = ConversationTurn(
            speaker="customer",
            text=message,
            timestamp=datetime.now()
        )
        conversation.conversation_history.append(customer_turn)
        
        try:
            # Prepare context for LLM
            context = await self._prepare_context(conversation)
            
            # Generate streaming response
            full_response = ""
            async for chunk in self.llm_provider.generate_stream(message, context):
                full_response += chunk
                yield chunk
            
            # Update conversation with agent response
            agent_turn = ConversationTurn(
                speaker="agent",
                text=full_response,
                timestamp=datetime.now()
            )
            conversation.conversation_history.append(agent_turn)
            
            # Process response for actions and context updates
            await self._process_response(conversation, full_response)
            
            # Update stage and extract information
            await self._update_conversation_context(conversation)
            
            # Track metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            track_metric("message_streamed", {
                "conversation_id": conversation_id,
                "processing_time": processing_time,
                "current_stage": conversation.current_stage.value,
                "message_length": len(message),
                "response_length": len(full_response)
            })
            
        except Exception as e:
            logger.exception("Error processing streaming message: %s", str(e))
            track_metric("stream_error", {
                "conversation_id": conversation_id,
                "error": str(e)
            })
            yield f"\n\nI apologize, but there was an error processing your message. Please try again."
    
    async def _prepare_context(self, conversation: ConversationContext) -> Dict[str, Any]:
        """
        Prepare the context for the LLM.
        
        Args:
            conversation: The conversation context.
            
        Returns:
            A dictionary containing all context needed for the LLM.
        """
        # Retrieve relevant knowledge
        product_info = await self.knowledge_base.get_products()
        company_info = await self.knowledge_base.get_company_info()
        
        # Retrieve relevant documents if available
        relevant_docs = []
        if conversation.conversation_history:
            last_message = conversation.conversation_history[-1].text
            relevant_docs = await self.knowledge_base.retrieve_relevant(last_message, limit=3)
        
        # Long-term memory retrieval
        memory_items = await self.memory_manager.retrieve_relevant(
            conversation_id=conversation.conversation_id,
            query=conversation.conversation_history[-1].text if conversation.conversation_history else ""
        )
        
        # Build the context
        context = {
            "conversation_id": conversation.conversation_id,
            "current_stage": conversation.current_stage,
            "previous_stage": conversation.previous_stage,
            "business_info": company_info,
            "products": product_info,
            "relevant_documents": relevant_docs,
            "memory_items": memory_items,
            "conversation_history": conversation.conversation_history[-10:],  # Last 10 turns for context window management
            "full_history_summary": await self.memory_manager.get_conversation_summary(conversation.conversation_id),
            "customer_info": {
                "name": conversation.customer_name,
                "email": conversation.customer_email,
                "company": conversation.customer_company,
                "role": conversation.customer_role,
            },
            "detected_needs": conversation.detected_needs,
            "detected_objections": conversation.detected_objections,
            "product_interests": conversation.product_interests
        }
        
        return context
    
    async def _process_response(self, conversation: ConversationContext, response: str) -> None:
        """
        Process the agent's response to extract actions and update context.
        
        Args:
            conversation: The conversation context.
            response: The agent's response text.
        """
        # Check for action triggers in the response
        if "schedule a meeting" in response.lower() or "book a time" in response.lower():
            # Extract potential email if available
            if conversation.customer_email:
                meeting_link = await self.action_engine.schedule_meeting(conversation.customer_email)
                conversation.follow_up_actions.append(f"Meeting scheduled: {meeting_link}")
        
        if "send you more information" in response.lower() or "email you" in response.lower():
            # If we have the email, queue an email action
            if conversation.customer_email:
                # Simplified - in reality would use a template and more context
                await self.action_engine.send_email(
                    to=conversation.customer_email,
                    subject="Information about our products",
                    body=f"Hello {conversation.customer_name or 'there'},\n\nThank you for your interest in our products. Here is the information you requested...",
                )
                conversation.follow_up_actions.append(f"Email sent to: {conversation.customer_email}")
        
        # Store response in memory
        await self.memory_manager.add_interaction(
            conversation_id=conversation.conversation_id,
            turn=conversation.conversation_history[-1]
        )
    
    async def _update_conversation_context(self, conversation: ConversationContext) -> None:
        """
        Update the conversation context with new information and potentially advance the sales stage.
        
        Args:
            conversation: The conversation context.
        """
        # In a production implementation, we would use another LLM call or 
        # a specialized model to analyze the conversation and extract information.
        # For simplicity, here we'll use a rule-based approach.
        
        # Extract customer information if not already set
        await self._extract_customer_info(conversation)
        
        # Determine if stage should change
        await self._update_sales_stage(conversation)
    
    async def _extract_customer_info(self, conversation: ConversationContext) -> None:
        """
        Extract customer information from conversation history.
        
        Args:
            conversation: The conversation context.
        """
        # In a real implementation, this would use NLP or another LLM call
        # Here we'll use a simplified approach for demonstration
        
        # Only check recent messages to avoid reprocessing the entire history
        recent_messages = conversation.conversation_history[-5:]
        for turn in recent_messages:
            if turn.speaker == "customer":
                text = turn.text.lower()
                
                # Very simplistic email extraction (would use regex or NER in production)
                if "@" in text and "." in text and not conversation.customer_email:
                    words = text.split()
                    for word in words:
                        if "@" in word and "." in word:
                            email = word.strip(".,;:!?")
                            conversation.customer_email = email
                            logger.info("Extracted customer email: %s", email)
                
                # Very simplistic name extraction (would use NER in production)
                if "my name is" in text and not conversation.customer_name:
                    name_part = text.split("my name is")[1].strip().split()[0]
                    conversation.customer_name = name_part
                    logger.info("Extracted customer name: %s", name_part)
                
                # Very simplistic company extraction (would use NER in production)
                if "work for" in text and not conversation.customer_company:
                    company_part = text.split("work for")[1].strip().split()[0]
                    conversation.customer_company = company_part
                    logger.info("Extracted customer company: %s", company_part)
    
    async def _update_sales_stage(self, conversation: ConversationContext) -> None:
        """
        Update the sales stage based on conversation progress.
        
        Args:
            conversation: The conversation context.
        """
        # In a real implementation, this would use another LLM call to analyze
        # the conversation and determine the appropriate stage
        # Here we'll use a simplified rule-based approach
        
        current_stage = conversation.current_stage
        
        # Get the last few exchanges
        if len(conversation.conversation_history) >= 2:
            last_agent_msg = ""
            last_customer_msg = ""
            
            # Find last agent and customer messages
            for turn in reversed(conversation.conversation_history):
                if turn.speaker == "agent" and not last_agent_msg:
                    last_agent_msg = turn.text.lower()
                elif turn.speaker == "customer" and not last_customer_msg:
                    last_customer_msg = turn.text.lower()
                
                if last_agent_msg and last_customer_msg:
                    break
            
            # Simple stage transition rules (would be more sophisticated in real implementation)
            if current_stage == SalesStage.INTRODUCTION:
                # After introduction and some customer info, move to qualification
                if len(conversation.conversation_history) >= 4:  # After 2 exchanges
                    conversation.previous_stage = current_stage
                    conversation.current_stage = SalesStage.QUALIFICATION
                    logger.info("Advancing from INTRODUCTION to QUALIFICATION stage")
            
            elif current_stage == SalesStage.QUALIFICATION:
                # If customer mentions problems, pain points, needs, move to needs analysis
                need_indicators = ["need", "problem", "issue", "challenge", "looking for", "trying to"]
                if any(indicator in last_customer_msg for indicator in need_indicators):
                    conversation.previous_stage = current_stage
                    conversation.current_stage = SalesStage.NEEDS_ANALYSIS
                    logger.info("Advancing from QUALIFICATION to NEEDS_ANALYSIS stage")
            
            elif current_stage == SalesStage.NEEDS_ANALYSIS:
                # If agent has presented solutions/products, move to solution presentation
                solution_indicators = ["recommend", "suggest", "offer", "solution", "product", "service"]
                if any(indicator in last_agent_msg for indicator in solution_indicators):
                    conversation.previous_stage = current_stage
                    conversation.current_stage = SalesStage.SOLUTION_PRESENTATION
                    logger.info("Advancing from NEEDS_ANALYSIS to SOLUTION_PRESENTATION stage")
            
            elif current_stage == SalesStage.SOLUTION_PRESENTATION:
                # If customer expresses concerns, move to objection handling
                objection_indicators = ["expensive", "costly", "concern", "worry", "not sure", "competition"]
                if any(indicator in last_customer_msg for indicator in objection_indicators):
                    conversation.previous_stage = current_stage
                    conversation.current_stage = SalesStage.OBJECTION_HANDLING
                    logger.info("Advancing from SOLUTION_PRESENTATION to OBJECTION_HANDLING stage")
            
            elif current_stage == SalesStage.OBJECTION_HANDLING:
                # If agent is discussing next steps, move to close
                close_indicators = ["next steps", "move forward", "schedule", "demo", "trial", "package", "pricing"]
                if any(indicator in last_agent_msg for indicator in close_indicators):
                    conversation.previous_stage = current_stage
                    conversation.current_stage = SalesStage.CLOSE
                    logger.info("Advancing from OBJECTION_HANDLING to CLOSE stage")
            
            # Check for conversation end signals regardless of current stage
            end_indicators = ["goodbye", "bye", "thank you", "thanks", "end", "not interested"]
            if any(indicator in last_customer_msg for indicator in end_indicators) and "can help" not in last_customer_msg:
                conversation.previous_stage = current_stage
                conversation.current_stage = SalesStage.END_CONVERSATION
                logger.info("Moving to END_CONVERSATION stage")
                
        # Track stage change
        if conversation.current_stage != current_stage:
            track_metric("stage_change", {
                "conversation_id": conversation.conversation_id,
                "previous_stage": current_stage.value,
                "new_stage": conversation.current_stage.value
            })