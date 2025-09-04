"""Text block agent for generating HTML content using Langfuse integration."""

import json
import logging
from datetime import datetime
from typing import Dict, Any

from langchain_core.agents import AgentFinish
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from agent.models import WidgetAgentState, TextBlockContentSchema
from config import get_langfuse_callback_handler, LANGFUSE_AVAILABLE
from .tools.fetch_widget import fetch_widget_tool

# Handle imports for different execution contexts
try:
    from actions.prompts import compile_prompt, get_prompt_config
    from actions.dashboard import update_widget
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.prompts import compile_prompt, get_prompt_config
    from actions.dashboard import update_widget

logger = logging.getLogger(__name__)


class TextBlockAgent:
    """Agent for generating text block content using Langfuse prompts and LLM configuration."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize text block agent with Langfuse configuration."""
        self.llm_model = llm_model
        self.agent_executor = None
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the react agent with Langfuse model configuration."""
        try:
            logger.info("Fetching model configuration from Langfuse for text block agent...")
            
            # Get model configuration from Langfuse (REQUIRED)
            prompt_config = get_prompt_config("widget_agent_team/text_block_agent", label="latest")
            
            # Extract required model and temperature from Langfuse config
            model = prompt_config.get("model")
            temperature = prompt_config.get("temperature")
            reasoning_effort = prompt_config.get("reasoning_effort")
            
            # Validate required configuration
            if not model:
                raise ValueError("Model configuration is missing or empty in Langfuse prompt config")
            if temperature is None:
                raise ValueError("Temperature configuration is missing in Langfuse prompt config")
            
            logger.info(f"✅ Using Langfuse model config - model: {model}, temperature: {temperature}, reasoning_effort: {reasoning_effort}")
            
            # Initialize LLM with Langfuse configuration
            llm_params = {
                "model": model,
                "temperature": temperature
            }
            
            # Add reasoning_effort if provided (for reasoning models like o1, o3, o4-mini)
            if reasoning_effort:
                llm_params["reasoning_effort"] = reasoning_effort
                
            llm = ChatOpenAI(**llm_params)
            
            # Create tools for the agent (removed fetch_widget_tool - now using reference_widget_data)
            tools = [self._create_generate_content_tool()]
            
            # Use fallback prompt - agent prompt should be simple for ReactAgent
            agent_prompt = """You are a text block content generator. Use the available tools to generate appropriate HTML content.
                
Your responsibilities:
1. Use the generate_text_content tool to create appropriate HTML content for text blocks
2. Use any provided reference widget data to create explanatory content
3. Consider the context and purpose of the text block when generating content
4. Ensure the content is well-formatted and informative

When calling generate_text_content tool, provide a content_request dictionary with these keys:
- widget_details: Any reference widget data provided in the input (or empty string if none)
- user_prompt: The user's original request
- task_instructions: The specific task instructions"""

            # Create react agent with proper parameters
            self.agent_executor = create_react_agent(
                model=llm,
                tools=tools,
                prompt=agent_prompt
            )
            
            logger.info("✅ Text block agent initialized successfully with Langfuse configuration")
            
        except Exception as e:
            error_msg = f"Failed to initialize TextBlockAgent - cannot fetch model config from Langfuse: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # _create_fetch_widget_tool removed - now using reference_widget_data passed directly in state

    def _create_generate_content_tool(self):
        """Create tool for generating text block content."""
        @tool
        def generate_text_content(content_request: dict) -> str:
            """Generate HTML content for text block widget using Langfuse prompt.
            
            Args:
                content_request: Dictionary containing:
                    - widget_details: Widget details as JSON string
                    - user_prompt: User's original request  
                    - task_instructions: Specific task instructions
                    
            Returns:
                Generated HTML content
            """
            try:
                # Extract parameters from content_request dictionary
                widget_details = content_request.get('widget_details', '{}')
                user_prompt = content_request.get('user_prompt', '')
                task_instructions = content_request.get('task_instructions', '')
                
                # Parse widget details
                try:
                    widget_data = json.loads(widget_details)
                except json.JSONDecodeError:
                    widget_data = {"config": {}, "title": "", "description": ""}
                
                # Prepare dynamic variables for Langfuse prompt
                prompt_variables = {
                    "user_prompt": user_prompt,
                    "task_instructions": task_instructions,
                    "widget_title": widget_data.get("title", ""),
                    "widget_description": widget_data.get("description", ""),
                    "widget_config": json.dumps(widget_data.get("config", {}), indent=2),
                    "widget_data": json.dumps(widget_data.get("data", {}), indent=2),
                    "current_timestamp": datetime.now().isoformat(),
                    "widget_id": widget_data.get("widget_id", ""),
                    "dashboard_id": widget_data.get("dashboard_id", "")
                }
                
                logger.info("Fetching and compiling text block prompt from Langfuse...")
                
                # Compile the prompt with dynamic variables from Langfuse (REQUIRED)
                text_prompt = compile_prompt(
                    "widget_agent_team/text_block_agent", 
                    prompt_variables,
                    label="latest"
                )
                
                # Validate compiled prompt
                if not text_prompt:
                    raise ValueError("Compiled text block prompt from Langfuse is empty or None")
                
                text_prompt_str = str(text_prompt)
                if not text_prompt_str or len(text_prompt_str.strip()) == 0:
                    raise ValueError("Compiled text block prompt from Langfuse is empty or invalid")
                
                logger.info(f"✅ Successfully compiled Langfuse text block prompt with {len(prompt_variables)} variables")
                
                # Get LLM configuration again for content generation
                prompt_config = get_prompt_config("widget_agent_team/text_block_agent", label="latest")
                model = prompt_config.get("model")
                temperature = prompt_config.get("temperature", 0.7)
                reasoning_effort = prompt_config.get("reasoning_effort")
                
                # Create LLM instance for content generation
                content_llm_params = {
                    "model": model,
                    "temperature": temperature
                }
                
                # Add reasoning_effort if provided (for reasoning models like o1, o3, o4-mini)
                if reasoning_effort:
                    content_llm_params["reasoning_effort"] = reasoning_effort
                    
                content_llm = ChatOpenAI(**content_llm_params)
                
                # Generate content with structured output
                structured_llm = content_llm.with_structured_output(TextBlockContentSchema)
                
                # Create Langfuse callback handler for text block content generation
                content_config = {}
                if LANGFUSE_AVAILABLE:
                    try:
                        langfuse_handler = get_langfuse_callback_handler(
                            trace_name="text-block-content-generation",
                            session_id=state.chat_id,
                            user_id=getattr(state, 'user_id', None),
                            tags=["text-block", "content-generation", "structured"],
                            metadata={
                                "dashboard_id": state.dashboard_id,
                                "widget_id": state.widget_id,
                                "operation": state.operation
                            }
                        )
                        if langfuse_handler:
                            content_config = {"callbacks": [langfuse_handler]}
                    except Exception as langfuse_error:
                        logger.warning(f"Failed to create Langfuse handler for text content generation: {langfuse_error}")
                
                # Invoke structured LLM with or without tracing
                if content_config:
                    response = structured_llm.invoke(text_prompt_str, config=content_config)
                else:
                    response = structured_llm.invoke(text_prompt_str)
                
                # Extract content from response
                if hasattr(response, 'content'):
                    generated_content = response.content
                else:
                    generated_content = str(response)
                
                logger.info("✅ Successfully generated text block content using Langfuse prompt")
                
                return generated_content
                
            except Exception as e:
                error_msg = f"Failed to generate text content: {str(e)}"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
        
        return generate_text_content

    def generate_text_block(self, state: WidgetAgentState) -> Command:
        """Generate text block content and update widget configuration.
        
        Args:
            state: Current widget agent state
            
        Returns:
            Command with updated state
        """
        try:
            if not self.agent_executor:
                raise RuntimeError("Agent not properly initialized")
            
            # Check if we have reference widget data from other completed tasks
            reference_context = ""
            reference_data_json = ""
            if state.reference_widget_data and len(state.reference_widget_data) > 0:
                reference_context = f"""

CRITICAL: This text block must analyze {len(state.reference_widget_data)} existing widget(s) that were created in previous tasks.
The widget data has been provided directly - NO NEED to fetch from database.
DO NOT use the current widget ID {state.widget_id} - that's the text block being created.
USE the reference widget data provided below to analyze and explain the widget(s)."""
                
                # Convert reference widget data to JSON for the prompt
                import json
                try:
                    reference_data_json = json.dumps(state.reference_widget_data, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"Failed to serialize reference widget data: {e}")
                    reference_data_json = str(state.reference_widget_data)
            
            # Prepare input for the agent
            agent_input = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Generate HTML content for a text block widget.
                        
User Request: {state.user_prompt}
Task Instructions: {state.task_instructions}
Widget ID: {state.widget_id}
{reference_context}

Use the available tools to:
{("1. Generate appropriate HTML content for the text block" if not state.reference_widget_data else f"1. Generate HTML content explaining the referenced widget(s) based on the provided data below\n\nREFERENCE WIDGET DATA:\n{reference_data_json}")}

Requirements:
- Use h2 tags for headings (not h1)
- Do not include any inline styles or styling attributes
- Generate clean, semantic HTML content
- Focus on the content described in the user request and task instructions
"""
                    }
                ]
            }
            
            logger.info(f"Starting text block generation for widget_id: {state.widget_id}")
            
            # Create Langfuse callback handler for text block agent execution
            agent_config = {}
            if LANGFUSE_AVAILABLE:
                try:
                    langfuse_handler = get_langfuse_callback_handler(
                        trace_name="text-block-agent-execution",
                        session_id=state.chat_id,
                        user_id=getattr(state, 'user_id', None),
                        tags=["text-block-agent", "react-agent", "execution"],
                        metadata={
                            "dashboard_id": state.dashboard_id,
                            "widget_id": state.widget_id,
                            "operation": state.operation,
                            "task_id": state.task_id
                        }
                    )
                    if langfuse_handler:
                        agent_config = {"callbacks": [langfuse_handler]}
                except Exception as langfuse_error:
                    logger.warning(f"Failed to create Langfuse handler for text block agent: {langfuse_error}")
            
            # Execute the agent with or without tracing
            if agent_config:
                result = self.agent_executor.invoke(agent_input, config=agent_config)
            else:
                result = self.agent_executor.invoke(agent_input)
            
            # Extract the generated content
            generated_content = ""
            if isinstance(result, dict):
                if 'messages' in result and result['messages']:
                    last_message = result['messages'][-1]
                    if hasattr(last_message, 'content'):
                        generated_content = last_message.content
                    else:
                        generated_content = str(last_message)
            elif isinstance(result, AgentFinish):
                generated_content = result.return_values.get('output', str(result))
            else:
                generated_content = str(result)
            
            # Create the unified widget configuration for validation and database persistence
            widget_config = {
                "content": generated_content
            }
            
            logger.info(f"✅ Successfully generated text block content for widget {state.widget_id}")
            
            # Return to widget_supervisor with widget_config (NOT generated_code for text blocks)
            return Command(
                goto="widget_supervisor",
                update={
                    "widget_config": widget_config,  # Unified config format for validation
                    "task_status": "in_progress",  # Still in progress, not completed
                    "updated_at": datetime.now(),
                }
            )
                
        except Exception as e:
            error_msg = f"Text block generation failed: {str(e)}"
            logger.error(error_msg)
            
            # Return to widget_supervisor even on error - let supervisor decide next step
            return Command(
                goto="widget_supervisor", 
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "task_status": "in_progress",  # Let supervisor decide if this is terminal
                    "updated_at": datetime.now(),
                }
            )


# Create lazy singleton instance
_text_block_agent_instance = None


def get_text_block_agent():
    """Get or create text block agent instance."""
    global _text_block_agent_instance
    if _text_block_agent_instance is None:
        _text_block_agent_instance = TextBlockAgent()
    return _text_block_agent_instance


def text_block_node(state: WidgetAgentState) -> Command:
    """Lazy wrapper for text block node."""
    return get_text_block_agent().generate_text_block(state)