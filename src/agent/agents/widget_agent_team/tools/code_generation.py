"""Python code generation tool for widget processing."""

import json
import os
import sys
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.models import WidgetAgentState

from actions.utils import get_chart_config_schema_string

# Handle imports for different execution contexts
try:
    from actions.prompts import compile_prompt, get_prompt_config
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.prompts import compile_prompt, get_prompt_config


@tool
def generate_python_code_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Generates Python code for processing widget data based on state requirements and file schemas."""
    try:
        # Extract required information from state
        widget_type = state.widget_type
        user_request = state.user_prompt
        operation = state.operation

        # Get schema information from state
        file_schemas = state.file_schemas or []
        file_sample_data = state.file_sample_data or []
        raw_file_data = state.raw_file_data or {}

        # Convert to simple format for prompt
        schemas_info = (
            [
                {
                    "file_id": schema.file_id,
                    "columns": [col.name for col in schema.columns],
                    "total_rows": schema.total_rows,
                }
                for schema in file_schemas
            ]
            if isinstance(file_schemas, list)
            else []
        )

        samples_info = (
            [
                {
                    "file_id": sample.file_id,
                    "headers": sample.headers,
                    "sample_rows": sample.sample_rows_returned,
                }
                for sample in file_sample_data
            ]
            if isinstance(file_sample_data, list)
            else []
        )

        # Get the schema string programmatically
        chart_config_schema = get_chart_config_schema_string()

        # Prepare dynamic variables for Langfuse prompt compilation
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # Prepare all 9 dynamic variables as specified by user
            prompt_variables = {
                "user_request": user_request,
                "widget_type": widget_type,
                "operation": operation,
                "file_schemas": json.dumps(schemas_info, indent=2) if schemas_info else "No schemas available",
                "sample_info": json.dumps(samples_info, indent=2) if samples_info else "No sample data available", 
                "chart_config_schema": chart_config_schema,
                "len_schemas_info": len(schemas_info),
                "schemas_info_rows": sum(schema.get("total_rows", 0) for schema in schemas_info),
                "schemas_info_columns_info": ", ".join([f"{schema.get('file_id', 'unknown')}: {len(schema.get('columns', []))} columns" for schema in schemas_info])
            }
            
            logger.info("Fetching and compiling code generation prompt from Langfuse...")
            
            # Compile the prompt with dynamic variables from Langfuse (REQUIRED)
            code_generation_prompt = compile_prompt(
                "widget_agent_team/data/tools/generate_python_code", 
                prompt_variables,
                label="latest"
            )
            
            # Validate compiled prompt (handle different formats)
            if not code_generation_prompt:
                raise ValueError("Compiled code generation prompt from Langfuse is empty or None")
            
            # Convert to string if needed and validate
            code_generation_prompt_str = str(code_generation_prompt)
            if not code_generation_prompt_str or len(code_generation_prompt_str.strip()) == 0:
                raise ValueError("Compiled code generation prompt from Langfuse is empty or invalid")
            
            logger.info(f"✅ Successfully compiled Langfuse code generation prompt with {len(prompt_variables)} variables")
            
            # Fetch model configuration from Langfuse (REQUIRED)
            prompt_config = get_prompt_config("widget_agent_team/data/tools/generate_python_code", label="latest")
            
            # Extract required model and temperature from Langfuse config
            model = prompt_config.get("model")
            temperature = prompt_config.get("temperature")
            
            # Validate required configuration
            if not model:
                raise ValueError("Model configuration is missing or empty in Langfuse prompt config")
            if temperature is None:
                raise ValueError("Temperature configuration is missing in Langfuse prompt config")
            
            logger.info(f"✅ Using Langfuse model config - model: {model}, temperature: {temperature}")
            
            # Create code generation LLM with Langfuse configuration
            code_gen_llm = ChatOpenAI(model=model, temperature=temperature)
            
            # Use the compiled prompt
            code_generation_prompt = code_generation_prompt_str
            
        except Exception as e:
            error_msg = f"Failed to fetch/compile code generation prompt from Langfuse: {str(e)}"
            import logging
            logging.getLogger(__name__).error(error_msg)
            return Command(
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                }
            )

        # Check if there are previous errors to include in the prompt for retry
        error_context = ""
        if state.error_messages:
            recent_errors = state.error_messages[
                -2:
            ]  # Include last 2 errors for context
            error_context = f"""

PREVIOUS ERRORS TO FIX:
{chr(10).join(recent_errors)}

Please fix these issues in your generated code.
"""

        final_prompt = code_generation_prompt + error_context
        generated_code = code_gen_llm.invoke(final_prompt).content

        # Clean up the code (remove markdown formatting if present)
        if "```python" in generated_code:
            generated_code = (
                generated_code.split("```python")[1].split("```")[0].strip()
            )
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        # Return Command with generated code
        success_msg = f"Generated Python code for {widget_type} widget:\n```python\n{generated_code}\n```"
        return Command(
            update={
                "generated_code": generated_code,
                "messages": [
                    ToolMessage(content=success_msg, tool_call_id=tool_call_id)
                ],
            }
        )

    except Exception as e:
        error_msg = f"Code generation failed: {str(e)}"
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )