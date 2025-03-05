"""
SalesGPT Configuration

This module defines the configuration classes and loading functions for AI Voice assited sales agent.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: str = "openai"  # openai, anthropic, huggingface, local
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    system_prompt_template: str = """
    You are a sales agent for {company_name}. You are an expert in sales and customer service.
    
    Company Information:
    - Company: {company_name}
    - Value Proposition: {value_proposition}
    
    Current sales stage: {current_stage}
    
    Available Products:
    {product_info}
    
    Instructions for this stage:
    {stage_guidance}
    
    Remember to be conversational, helpful, and focused on understanding and solving the customer's needs.
    Always be truthful about product capabilities and pricing.
    """

class KnowledgeBaseConfig(BaseModel):
    """Configuration for knowledge base"""
    type: str = "document"  # document, vector
    path: str = "data/knowledge"
    use_embeddings: bool = True
    embeddings_model: str = "openai"  # openai, huggingface
    embedding_dimension: int = 1536

class MemoryConfig(BaseModel):
    """Configuration for memory management"""
    use_summarization: bool = True
    turns_before_summarization: int = 10
    use_embeddings: bool = True
    embeddings_model: str = "openai"  # openai, huggingface
    hf_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    use_openai_summarization: bool = True
    openai_api_key: Optional[str] = None
    max_memory_items: int = 100  # Maximum items to keep in memory per conversation

class ActionsConfig(BaseModel):
    """Configuration for action engine"""
    email_provider: str = "smtp"  # smtp, sendgrid
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    sendgrid_api_key: Optional[str] = None
    
    calendar_provider: str = "calendly"  # google, calendly
    calendly_api_key: Optional[str] = None
    calendly_user: Optional[str] = None
    google_credentials_path: Optional[str] = None
    
    crm_provider: Optional[str] = None  # salesforce, hubspot
    crm_api_key: Optional[str] = None
    crm_base_url: Optional[str] = None

class WebConfig(BaseModel):
    """Configuration for web interface"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    enable_swagger: bool = True
    cors_origins: List[str] = ["*"]
    session_timeout_minutes: int = 60

class VoiceConfig(BaseModel):
    """Configuration for voice interface"""
    host: str = "0.0.0.0"
    port: int = 8001
    reload: bool = False
    
    # ASR (Automatic Speech Recognition) settings
    asr_provider: str = "whisper"  # whisper, google, azure, deepgram, local
    asr_api_key: Optional[str] = None
    
    # TTS (Text-to-Speech) settings
    tts_provider: str = "google"  # google, azure, elevenlabs, polly, openai
    tts_api_key: Optional[str] = None
    voice_id: str = "en-US-Neural2-F"  # Default female voice
    
    # Voice settings
    language: str = "en-US"
    sample_rate: int = 16000
    encoding: str = "LINEAR16"
    
    # Call settings
    auto_greeting: bool = True
    greeting_text: str = "Hello, this is SalesGPT. How can I help you today?"
    
    # Latency optimization
    enable_streaming: bool = True
    chunk_size: int = 4096
    max_silence_ms: int = 500
    vad_threshold: float = 0.3  # Voice Activity Detection threshold

class SecurityConfig(BaseModel):
    """Configuration for security"""
    enable_authentication: bool = False
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    admin_username: Optional[str] = None
    admin_password_hash: Optional[str] = None

class LoggingConfig(BaseModel):
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file: Optional[str] = None
    rotation_size_mb: int = 10
    max_log_files: int = 5

class SalesGPTConfig(BaseModel):
    """Main configuration class for SalesGPT"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    twilio: TwilioConfig = Field(default_factory=TwilioConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Additional configuration
    environment: str = "development"  # development, staging, production
    metrics_enabled: bool = True
    version: str = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)

def load_config(config_path: str) -> SalesGPTConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Loaded configuration object
    """
    try:
        # Check if the file exists
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found at {config_path}, using default configuration")
            return SalesGPTConfig()
        
        # Load the configuration file
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create the configuration object
        config = SalesGPTConfig(**config_data)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.info("Using default configuration")
        return SalesGPTConfig()

def save_config(config: SalesGPTConfig, config_path: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration object
        config_path: Path where to save the configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Convert to dict and save
        config_dict = config.model_dump()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False