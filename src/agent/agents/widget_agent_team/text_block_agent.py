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

from agent.models import WidgetAgentState
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
            
            # Validate required configuration
            if not model:
                raise ValueError("Model configuration is missing or empty in Langfuse prompt config")
            if temperature is None:
                raise ValueError("Temperature configuration is missing in Langfuse prompt config")
            
            logger.info(f"✅ Using Langfuse model config - model: {model}, temperature: {temperature}")
            
            # Initialize LLM with Langfuse configuration
            llm = ChatOpenAI(model=model, temperature=temperature)
            
            # Create tools for the agent
            tools = [self._create_fetch_widget_tool(), self._create_generate_content_tool()]
            
            # Create react agent
            self.agent_executor = create_react_agent(
                llm, 
                tools,
                state_modifier="You are a text block content generator. Use the available tools to fetch widget details and generate appropriate HTML content."
            )
            
            logger.info("✅ Text block agent initialized successfully with Langfuse configuration")
            
        except Exception as e:
            error_msg = f"Failed to initialize TextBlockAgent - cannot fetch model config from Langfuse: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _create_fetch_widget_tool(self):
        """Create tool for fetching widget details."""
        @tool
        def fetch_widget_details(widget_id: str) -> str:
            """Fetch widget details including config and metadata.
            
            Args:
                widget_id: The widget identifier
                
            Returns:
                Widget details as formatted string
            """
            try:
                from .tools.fetch_widget import fetch_widget_details
                details = fetch_widget_details(widget_id)
                return json.dumps(details, indent=2)
            except Exception as e:
                return f"ERROR: Failed to fetch widget details: {str(e)}"
        
        return fetch_widget_details

    def _create_generate_content_tool(self):
        """Create tool for generating text block content."""
        @tool
        def generate_text_content(widget_details: str, user_prompt: str, task_instructions: str) -> str:
            """Generate HTML content for text block widget using Langfuse prompt.
            
            Args:
                widget_details: Widget details as JSON string
                user_prompt: User's original request
                task_instructions: Specific task instructions
                
            Returns:
                Generated HTML content
            """
            try:
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
                
                # Create LLM instance for content generation
                content_llm = ChatOpenAI(model=model, temperature=temperature)
                
                # Generate content
                response = content_llm.invoke(text_prompt_str)
                
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
            
            # Prepare input for the agent
            agent_input = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Generate HTML content for a text block widget.
                        
User Request: {state.user_prompt}
Task Instructions: {state.task_instructions}
Widget ID: {state.widget_id}

Use the available tools to:
1. Fetch the current widget details and configuration
2. Generate appropriate HTML content for the text block

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
            
            # Execute the agent
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
            
            # Create the widget configuration
            widget_config = {
                "content": generated_content
            }
            
            # Update the widget in the database
            logger.info(f"Updating widget {state.widget_id} with generated text block content")
            
            update_result = update_widget(
                widget_id=state.widget_id,
                config=widget_config,
                is_configured=True
            )
            
            if update_result:
                logger.info(f"✅ Successfully updated text block widget {state.widget_id}")
                
                return Command(
                    goto="end",
                    update={
                        "generated_code": generated_content,
                        "widget_config": widget_config,
                        "widget_update_completed": True,
                        "task_status": "completed",
                        "updated_at": datetime.now(),
                    }
                )
            else:
                error_msg = "Failed to update widget in database"
                logger.error(error_msg)
                
                return Command(
                    goto="end",
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "task_status": "failed",
                        "updated_at": datetime.now(),
                    }
                )
                
        except Exception as e:
            error_msg = f"Text block generation failed: {str(e)}"
            logger.error(error_msg)
            
            return Command(
                goto="end", 
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "task_status": "failed",
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