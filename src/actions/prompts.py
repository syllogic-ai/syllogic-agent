"""Langfuse Prompt Management actions.

This module provides helper functions for managing prompts through Langfuse,
including retrieving, compiling, and managing prompt versions.
"""

import os
from typing import Any, Dict, Optional, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

# Handle imports for different execution contexts
try:
    from config import get_langfuse_client, get_prompt
except ImportError:
    import sys
    # Add the src directory to the path
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from config import get_langfuse_client, get_prompt

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
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


def get_prompt_config(prompt_name: str, default_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Get the configuration object for a prompt with fallback to default config.

    Args:
        prompt_name: Name of the prompt to retrieve
        default_config: Default configuration to use if prompt retrieval fails
        **kwargs: Additional arguments for prompt retrieval (version, label)

    Returns:
        Dictionary containing the prompt's configuration (model, temperature, etc.)
    """
    try:
        prompt = retrieve_prompt(prompt_name, **kwargs)
        config = prompt.config if hasattr(prompt, 'config') else {}
        
        logger.info(f"Retrieved config for prompt '{prompt_name}': {list(config.keys()) if config else 'No config'}")
        return config
        
    except Exception as e:
        logger.warning(f"Failed to get config for prompt '{prompt_name}', using default config: {str(e)}")
        return default_config or {
            "model": "gpt-4o-mini",
            "temperature": 0.3
        }


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


def extract_messages_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all messages from the previous steps in an agent's state.
    
    This function processes the messages field from agent states and returns
    a structured list of all messages (Human, AI, Tool) with their content
    and metadata in a consistent format.
    
    Args:
        state: Agent state dictionary containing a 'messages' field with BaseMessage objects
        
    Returns:
        List of dictionaries containing message information with keys:
        - role: Message role ('human', 'ai', 'tool', 'system')
        - content: Message content as string
        - message_type: Type of message (e.g., 'chat', 'tool-usage', 'task-list')
        - tool_calls: Tool calls if present (for AI messages)
        - tool_call_id: Tool call ID (for Tool messages)
        - additional_kwargs: Any additional metadata from the message
        
    Example:
        >>> from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        >>> state = {
        ...     'messages': [
        ...         HumanMessage(content="Create a bar chart"),
        ...         AIMessage(content="I'll help you create a bar chart", tool_calls=[...]),
        ...         ToolMessage(content="Chart created successfully", tool_call_id="call_123")
        ...     ]
        ... }
        >>> messages = extract_messages_from_state(state)
        >>> print(messages[0])
        {'role': 'human', 'content': 'Create a bar chart', 'message_type': 'chat', ...}
        
        # Format for use in prompts
        >>> formatted = format_messages_for_prompt(messages)
        >>> print(formatted)
        1. [HUMAN]: Create a bar chart
        2. [AI]: I'll help you create a bar chart [Tool calls: get_available_data()]
        3. [TOOL] (tool-usage): Chart created successfully [Tool call ID: call_123]
    """
    try:
        if hasattr(state, 'messages'):
            # Pydantic model - access messages directly
            messages = state.messages
        elif isinstance(state, dict):
            # Dictionary - use get method
            messages = state.get('messages', [])
        else:
            logger.warning(f"Unknown state type: {type(state)}")
            return []
            
        if not messages:
            logger.info("No messages found in agent state")
            return []
        
        extracted_messages = []

        for message in messages:
            # Handle different message types
            if isinstance(message, HumanMessage):
                message_data = {
                    'role': 'human',
                    'content': message.content,
                    'message_type': 'chat',
                    'additional_kwargs': message.additional_kwargs
                }
                
            elif isinstance(message, AIMessage):
                message_data = {
                    'role': 'ai',
                    'content': message.content,
                    'message_type': 'chat',
                    'tool_calls': getattr(message, 'tool_calls', []),
                    'additional_kwargs': message.additional_kwargs
                }
                
            elif isinstance(message, ToolMessage):
                message_data = {
                    'role': 'tool',
                    'content': message.content,
                    'message_type': 'tool-usage',
                    'tool_call_id': getattr(message, 'tool_call_id', None),
                    'additional_kwargs': message.additional_kwargs
                }
                
            else:
                # Handle other BaseMessage types or custom message types
                message_data = {
                    'role': getattr(message, 'type', 'unknown').lower().replace('message', ''),
                    'content': getattr(message, 'content', str(message)),
                    'message_type': 'chat',
                    'additional_kwargs': getattr(message, 'additional_kwargs', {})
                }
                
                # Try to extract tool calls if present
                if hasattr(message, 'tool_calls'):
                    message_data['tool_calls'] = message.tool_calls
                if hasattr(message, 'tool_call_id'):
                    message_data['tool_call_id'] = message.tool_call_id
                if hasattr(message, 'message_type'):
                    message_data['message_type'] = message.message_type
            
            extracted_messages.append(message_data)
        
        logger.info(f"Extracted {len(extracted_messages)} messages from agent state")
        return extracted_messages
        
    except Exception as e:
        logger.error(f"Failed to extract messages from state: {str(e)}")
        raise


def format_messages_for_prompt(messages: List[Dict[str, Any]], include_metadata: bool = False) -> str:
    """Format extracted messages into a readable string for use in prompts.
    
    Args:
        messages: List of message dictionaries from extract_messages_from_state
        include_metadata: Whether to include additional metadata in the output
        
    Returns:
        Formatted string representation of the message history
    """
    try:
        if not messages:
            return "No previous messages found."
        
        formatted_lines = []
        
        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            message_type = msg.get('message_type', 'chat')
            
            # Format the basic message
            line = f"{i}. [{role}]"
            if message_type != 'chat':
                line += f" ({message_type})"
            line += f": {content}"
            
            # Add tool calls if present
            if 'tool_calls' in msg and msg['tool_calls']:
                tool_calls_str = ", ".join([f"{tc.get('name', 'unknown')}()" for tc in msg['tool_calls']])
                line += f" [Tool calls: {tool_calls_str}]"
            
            # Add tool call ID if present
            if 'tool_call_id' in msg and msg['tool_call_id']:
                line += f" [Tool call ID: {msg['tool_call_id']}]"
            
            formatted_lines.append(line)
            
            # Add metadata if requested
            if include_metadata and msg.get('additional_kwargs'):
                metadata_str = ", ".join([f"{k}: {v}" for k, v in msg['additional_kwargs'].items()])
                formatted_lines.append(f"   Metadata: {metadata_str}")
        
        result = "\n".join(formatted_lines)
        logger.debug(f"Formatted {len(messages)} messages for prompt")
        return result
        
    except Exception as e:
        logger.error(f"Failed to format messages for prompt: {str(e)}")
        raise