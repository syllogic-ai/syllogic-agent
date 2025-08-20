"""Langfuse Prompt Management actions.

This module provides helper functions for managing prompts through Langfuse,
including retrieving, compiling, and managing prompt versions.
"""

import logging
from typing import Any, Dict, Optional

# Handle imports for different execution contexts
try:
    from config import get_langfuse_client, get_prompt
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config import get_langfuse_client, get_prompt

logger = logging.getLogger(__name__)


def retrieve_prompt(
    prompt_name: str,
    version: Optional[int] = None,
    label: Optional[str] = None
):
    """Retrieve a prompt from Langfuse.

    Args:
        prompt_name: Name of the prompt to retrieve
        version: Specific version number (optional)
        label: Specific label to retrieve (optional, e.g., "production")

    Returns:
        Langfuse prompt object with prompt content and configuration

    Raises:
        Exception: If prompt retrieval fails
    """
    try:
        return get_prompt(prompt_name, version=version, label=label)
    except Exception as e:
        logger.error(f"Failed to retrieve prompt '{prompt_name}': {str(e)}")
        raise


def compile_prompt(prompt_name: str, variables: Dict[str, Any], **kwargs) -> str:
    """Retrieve and compile a prompt with variables.

    Args:
        prompt_name: Name of the prompt to retrieve
        variables: Dictionary of variables to substitute in the prompt
        **kwargs: Additional arguments for prompt retrieval (version, label)

    Returns:
        Compiled prompt string with variables substituted

    Raises:
        Exception: If prompt retrieval or compilation fails
    """
    try:
        prompt = retrieve_prompt(prompt_name, **kwargs)
        compiled_prompt = prompt.compile(**variables)
        
        logger.info(f"Compiled prompt '{prompt_name}' with variables: {list(variables.keys())}")
        return compiled_prompt
        
    except Exception as e:
        logger.error(f"Failed to compile prompt '{prompt_name}': {str(e)}")
        raise


def get_prompt_config(prompt_name: str, **kwargs) -> Dict[str, Any]:
    """Get the configuration object for a prompt.

    Args:
        prompt_name: Name of the prompt to retrieve
        **kwargs: Additional arguments for prompt retrieval (version, label)

    Returns:
        Dictionary containing the prompt's configuration (model, temperature, etc.)

    Raises:
        Exception: If prompt retrieval fails
    """
    try:
        prompt = retrieve_prompt(prompt_name, **kwargs)
        config = prompt.config if hasattr(prompt, 'config') else {}
        
        logger.info(f"Retrieved config for prompt '{prompt_name}': {list(config.keys()) if config else 'No config'}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to get config for prompt '{prompt_name}': {str(e)}")
        raise


def create_prompt(
    name: str,
    prompt: str,
    config: Optional[Dict[str, Any]] = None,
    labels: Optional[list] = None,
    is_active: bool = False
):
    """Create a new prompt in Langfuse.

    Args:
        name: Name for the prompt
        prompt: The prompt content/template
        config: Configuration dictionary (model, temperature, etc.)
        labels: List of labels for the prompt
        is_active: Whether to immediately activate this prompt

    Returns:
        Created prompt object

    Raises:
        Exception: If prompt creation fails
    """
    try:
        langfuse_client = get_langfuse_client()
        
        created_prompt = langfuse_client.create_prompt(
            name=name,
            prompt=prompt,
            config=config or {},
            labels=labels or [],
            is_active=is_active
        )
        
        logger.info(f"Created prompt '{name}' with labels: {labels}")
        return created_prompt
        
    except Exception as e:
        logger.error(f"Failed to create prompt '{name}': {str(e)}")
        raise


def get_prompt_with_fallback(
    prompt_name: str,
    fallback_prompt: str,
    variables: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """Retrieve and compile a prompt with fallback to a default prompt.

    Args:
        prompt_name: Name of the prompt to retrieve from Langfuse
        fallback_prompt: Default prompt to use if retrieval fails
        variables: Variables to substitute in the prompt
        **kwargs: Additional arguments for prompt retrieval (version, label)

    Returns:
        Compiled prompt string (from Langfuse or fallback)
    """
    try:
        # Try to get prompt from Langfuse
        if variables:
            return compile_prompt(prompt_name, variables, **kwargs)
        else:
            prompt = retrieve_prompt(prompt_name, **kwargs)
            if hasattr(prompt, 'prompt'):
                prompt_content = prompt.prompt
                # Handle different prompt formats (string or chat messages)
                if isinstance(prompt_content, list):
                    return "\n".join([msg.get('content', str(msg)) for msg in prompt_content])
                else:
                    return str(prompt_content)
            else:
                return str(prompt)
            
    except Exception as e:
        logger.warning(f"Failed to retrieve prompt '{prompt_name}', using fallback: {str(e)}")
        
        # Use fallback prompt
        if variables:
            # Simple string formatting for fallback
            try:
                return fallback_prompt.format(**variables)
            except KeyError as ke:
                logger.warning(f"Missing variable in fallback prompt: {ke}")
                return fallback_prompt
        else:
            return fallback_prompt


def list_prompts() -> list:
    """List all available prompts in Langfuse.

    Returns:
        List of prompt metadata

    Raises:
        Exception: If listing prompts fails
    """
    try:
        langfuse_client = get_langfuse_client()
        prompts = langfuse_client.api.prompts.list()
        
        logger.info(f"Retrieved {len(prompts.data) if prompts.data else 0} prompts from Langfuse")
        return prompts.data if prompts.data else []
        
    except Exception as e:
        logger.error(f"Failed to list prompts: {str(e)}")
        raise