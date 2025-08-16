"""Configuration module for the agent system.

This module initializes and manages shared resources like the Supabase client,
making them available throughout the application without passing as arguments.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from supabase import Client, create_client

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Global Supabase client instance
_supabase_client: Optional[Client] = None

# Global E2B sandbox configuration
_e2b_api_key: Optional[str] = None


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


def create_e2b_sandbox() -> Sandbox:
    """Create a new E2B Sandbox instance.

    Returns:
        Sandbox: New E2B Sandbox instance

    Raises:
        ValueError: If E2B API key is not available
        Exception: If sandbox creation fails
    """
    try:
        api_key = get_e2b_api_key()

        # Create Sandbox with API key
        sandbox = Sandbox(api_key=api_key)

        logger.info("E2B Sandbox created successfully")
        return sandbox

    except Exception as e:
        logger.error(f"Failed to create E2B Sandbox: {str(e)}")
        raise


def reset_e2b_config():
    """Reset the global E2B configuration.

    Useful for testing or when credentials change.
    """
    global _e2b_api_key
    _e2b_api_key = None
    logger.info("E2B configuration reset")


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


# Export public functions
__all__ = [
    "get_supabase_client",
    "reset_supabase_client",
    "get_e2b_api_key",
    "create_e2b_sandbox",
    "reset_e2b_config",
]
