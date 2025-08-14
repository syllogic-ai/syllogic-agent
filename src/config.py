"""Configuration module for the agent system.

This module initializes and manages shared resources like the Supabase client,
making them available throughout the application without passing as arguments.
"""

import os
import logging
from typing import Optional
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# Global Supabase client instance
_supabase_client: Optional[Client] = None


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
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not supabase_key:
            raise ValueError("Either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY environment variable is required")
        
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


# Initialize client on module import
try:
    get_supabase_client()
except Exception as e:
    logger.warning(f"Could not initialize Supabase client on import: {str(e)}")
    # Don't raise here - let functions handle the error when they try to use the client


# Export public functions
__all__ = ["get_supabase_client", "reset_supabase_client"]