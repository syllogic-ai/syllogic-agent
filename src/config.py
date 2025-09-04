"""Configuration module for the agent system.

This module initializes and manages shared resources like the Supabase client,
making them available throughout the application without passing as arguments.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client

# Import langfuse conditionally to avoid breaking the system when not available
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    CallbackHandler = None
    LANGFUSE_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Global Supabase client instance
_supabase_client: Optional[Client] = None

# Global E2B sandbox configuration
_e2b_api_key: Optional[str] = None

# Global Langfuse client instance
_langfuse_client: Optional["Langfuse"] = None


def get_supabase_client() -> Client:
    """Get or create the Supabase client instance.

    Returns:
        Client: Initialized Supabase client

    Raises:
        ValueError: If required environment variables are not set
        Exception: If client initialization fails
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    try:
        # Get environment variables
        supabase_url = os.getenv("SUPABASE_URL")

        # Try SERVICE_KEY first (preferred for server-side operations), then ANON_KEY
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv(
            "SUPABASE_ANON_KEY"
        )

        if not supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not supabase_key:
            raise ValueError(
                "Either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY environment variable is required"
            )

        # Create client
        _supabase_client = create_client(supabase_url, supabase_key)

        logger.info("Supabase client initialized successfully")
        return _supabase_client

    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        raise


def reset_supabase_client():
    """Reset the global Supabase client instance.

    Useful for testing or when credentials change.
    """
    global _supabase_client
    _supabase_client = None
    logger.info("Supabase client reset")


def get_e2b_api_key() -> str:
    """Get E2B API key from environment variables.

    Returns:
        str: E2B API key

    Raises:
        ValueError: If E2B_API_KEY environment variable is not set
    """
    global _e2b_api_key

    if _e2b_api_key is not None:
        return _e2b_api_key

    try:
        # Get E2B API key from environment
        # Try both E2B_SANDBOX_API_KEY (custom) and E2B_API_KEY (standard)
        api_key = os.getenv("E2B_SANDBOX_API_KEY") or os.getenv("E2B_API_KEY")

        if not api_key:
            raise ValueError("E2B_SANDBOX_API_KEY or E2B_API_KEY environment variable is required")

        _e2b_api_key = api_key
        logger.info("E2B API key loaded successfully")
        return _e2b_api_key

    except Exception as e:
        logger.error(f"Failed to load E2B API key: {str(e)}")
        raise



def reset_e2b_config():
    """Reset the global E2B configuration.

    Useful for testing or when credentials change.
    """
    global _e2b_api_key
    _e2b_api_key = None
    logger.info("E2B configuration reset")


def get_langfuse_client() -> Optional["Langfuse"]:
    """Get or create the Langfuse client instance.

    Returns:
        Langfuse: Initialized Langfuse client or None if langfuse not available

    Raises:
        ValueError: If required environment variables are not set
        Exception: If client initialization fails
    """
    global _langfuse_client

    if not LANGFUSE_AVAILABLE:
        raise ImportError("Langfuse is not available. Install with: pip install langfuse")

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        # Get environment variables
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not langfuse_secret_key:
            raise ValueError("LANGFUSE_SECRET_KEY environment variable is required")
        if not langfuse_public_key:
            raise ValueError("LANGFUSE_PUBLIC_KEY environment variable is required")

        # Create client
        _langfuse_client = Langfuse(
            secret_key=langfuse_secret_key,
            public_key=langfuse_public_key,
            host=langfuse_host
        )

        # Verify authentication
        if _langfuse_client.auth_check():
            logger.info("Langfuse client initialized and authenticated successfully")
        else:
            logger.warning("Langfuse client initialized but authentication check failed")

        return _langfuse_client

    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {str(e)}")
        raise


def reset_langfuse_client():
    """Reset the global Langfuse client instance.

    Useful for testing or when credentials change.
    """
    global _langfuse_client
    _langfuse_client = None
    logger.info("Langfuse client reset")


def get_prompt(prompt_name: str, version: Optional[int] = None, label: Optional[str] = None):
    """Get a prompt from Langfuse.

    Args:
        prompt_name: Name of the prompt to retrieve
        version: Specific version number (optional)
        label: Specific label to retrieve (optional, e.g., "production")

    Returns:
        Langfuse prompt object

    Raises:
        Exception: If prompt retrieval fails or langfuse not available
    """
    try:
        if not LANGFUSE_AVAILABLE:
            raise ImportError("Langfuse is not available. Install with: pip install langfuse")

        langfuse_client = get_langfuse_client()
        
        if version is not None:
            prompt = langfuse_client.get_prompt(prompt_name, version=version)
        elif label is not None:
            prompt = langfuse_client.get_prompt(prompt_name, label=label)
        else:
            prompt = langfuse_client.get_prompt(prompt_name)
        
        logger.info(f"Retrieved prompt '{prompt_name}' successfully")
        return prompt

    except Exception as e:
        logger.error(f"Failed to retrieve prompt '{prompt_name}': {str(e)}")
        raise


def get_langfuse_callback_handler(
    trace_name: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None
) -> Optional["CallbackHandler"]:
    """Get a Langfuse callback handler for LangChain tracing.
    
    Args:
        trace_name: Optional name for the trace (passed via langchain config metadata)
        session_id: Optional session ID for grouping traces (passed via langchain config metadata)
        user_id: Optional user ID for tracking (passed via langchain config metadata)
        tags: Optional list of tags for the trace (passed via langchain config metadata)  
        metadata: Optional metadata dictionary (passed via langchain config metadata)
    
    Returns:
        CallbackHandler: Initialized Langfuse callback handler or None if unavailable
        
    Note:
        Trace attributes should be set via the LangChain config metadata when invoking,
        not through the handler constructor. The handler itself is created without parameters.
    """
    try:
        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse is not available - skipping tracing")
            return None
            
        langfuse_client = get_langfuse_client()
        if not langfuse_client:
            logger.warning("Langfuse client not available - skipping tracing")
            return None
            
        # Create basic callback handler - trace attributes will be set via config metadata
        handler = CallbackHandler()
            
        logger.info("Langfuse callback handler created successfully")
        return handler
        
    except Exception as e:
        logger.error(f"Failed to create Langfuse callback handler: {str(e)}")
        # Return None instead of raising to avoid breaking the system
        return None


def get_langchain_config_with_tracing(
    trace_name: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None
) -> dict:
    """Get a LangChain config dict with Langfuse tracing enabled.
    
    Args:
        trace_name: Optional name for the trace
        session_id: Optional session ID for grouping traces
        user_id: Optional user ID for tracking
        tags: Optional list of tags for the trace
        metadata: Optional metadata dictionary
    
    Returns:
        Dict: LangChain config with callbacks and metadata for Langfuse tracing
    """
    try:
        if not LANGFUSE_AVAILABLE:
            return {}
            
        handler = get_langfuse_callback_handler()
        if not handler:
            return {}
            
        # Build metadata for trace attributes
        trace_metadata = metadata.copy() if metadata else {}
        if session_id:
            trace_metadata['langfuse_session_id'] = session_id
        if user_id:
            trace_metadata['langfuse_user_id'] = user_id
        if tags:
            trace_metadata['langfuse_tags'] = tags
            
        config = {
            "callbacks": [handler],
            "metadata": trace_metadata
        }
        
        # Add run_name for trace naming
        if trace_name:
            config["run_name"] = trace_name
            
        return config
        
    except Exception as e:
        logger.error(f"Failed to create Langchain config with tracing: {str(e)}")
        return {}


def create_langfuse_config(state: dict, trace_name: str = "syllogic-agent-execution") -> dict:
    """Create Langfuse configuration for tracing based on state context.
    
    This helper function extracts context from graph state and creates proper
    Langfuse tracing configuration for LangGraph execution.
    
    Args:
        state: The current graph state containing user/session info
        trace_name: Name for the trace
        
    Returns:
        Dict containing callbacks configuration or empty dict if Langfuse unavailable
    """
    if not LANGFUSE_AVAILABLE:
        return {}
        
    try:
        # Extract context information from state
        user_id = state.get("user_id")
        chat_id = state.get("chat_id") 
        dashboard_id = state.get("dashboard_id")
        request_id = state.get("request_id")
        
        # Create trace metadata
        metadata = {
            "dashboard_id": dashboard_id,
            "request_id": request_id,
        }
        
        # Use the existing helper function
        config = get_langchain_config_with_tracing(
            trace_name=trace_name,
            session_id=chat_id,  # Use chat_id as session for conversation grouping
            user_id=user_id,
            tags=["syllogic", "agent", "langgraph", "multi-agent"],
            metadata=metadata
        )
        
        if config:
            logger.info(f"Created Langfuse tracing for user={user_id}, chat={chat_id}, dashboard={dashboard_id}")
            
        return config
            
    except Exception as e:
        logger.error(f"Error creating Langfuse config: {e}")
        return {}


# Initialize client on module import
try:
    get_supabase_client()
except Exception as e:
    logger.warning(f"Could not initialize Supabase client on import: {str(e)}")
    # Don't raise here - let functions handle the error when they try to use the client

# Initialize E2B configuration on module import
try:
    get_e2b_api_key()
except Exception as e:
    logger.warning(f"Could not initialize E2B API key on import: {str(e)}")
    # Don't raise here - let functions handle the error when they try to use E2B

# Initialize Langfuse client on module import (only if available)
if LANGFUSE_AVAILABLE:
    try:
        get_langfuse_client()
    except Exception as e:
        logger.warning(f"Could not initialize Langfuse client on import: {str(e)}")
        # Don't raise here - let functions handle the error when they try to use Langfuse
else:
    logger.info("Langfuse not available - skipping initialization")


# Export public functions
__all__ = [
    "get_supabase_client",
    "reset_supabase_client",
    "get_e2b_api_key",
    "reset_e2b_config",
    "get_langfuse_client",
    "reset_langfuse_client",
    "get_prompt",
    "get_langfuse_callback_handler",
    "get_langchain_config_with_tracing",
    "create_langfuse_config",
    "LANGFUSE_AVAILABLE",
]
