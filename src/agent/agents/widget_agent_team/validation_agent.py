"""Data validation agent for widget processing."""

import json
import logging
from datetime import datetime
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.models import WidgetAgentState

# Initialize logger
logger = logging.getLogger(__name__)

# Handle imports for different execution contexts
try:
    from actions.prompts import compile_prompt, get_prompt_config
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.prompts import compile_prompt, get_prompt_config


class DataValidationResult(BaseModel):
    """Structured output for data validation assessment"""

    is_valid: bool = Field(
        description="Whether the generated data meets the requirements"
    )
    confidence_level: int = Field(
        description="Confidence level from 0-100", ge=0, le=100
    )
    explanation: str = Field(description="Detailed explanation of validation result")
    missing_requirements: Optional[List[str]] = Field(
        default=None, description="List of missing requirements if validation fails"
    )
    data_quality_issues: Optional[List[str]] = Field(
        default=None, description="List of data quality issues identified"
    )


class ValidationAgent:
    """LLM-based validation agent for widget data quality assessment."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize validation agent with LLM configuration from Langfuse."""
        # Fetch model configuration from Langfuse (REQUIRED)
        try:
            logger.info("Fetching model configuration from Langfuse for validate_data...")
            prompt_config = get_prompt_config("widget_agent_team/validate_data", label="latest")
            
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
            
            # Store configuration for validation operations
            self.model = model
            self.temperature = temperature
            self.reasoning_effort = reasoning_effort
            
        except Exception as e:
            error_msg = f"Failed to initialize ValidationAgent - cannot fetch model config from Langfuse: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def validate_data(self, state: WidgetAgentState) -> Command:
        """LLM-based validation that analyzes generated data against user requirements."""

        try:
            # Check if we have widget config ready for validation
            if state.widget_config is None:
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages
                        + ["No widget configuration to validate"],
                        "data_validated": False,
                        "updated_at": datetime.now(),
                    },
                )

            # For backwards compatibility, still check for execution errors if code_execution_result exists
            # This handles cases where data node produced errors
            if (
                state.code_execution_result is not None
                and isinstance(state.code_execution_result, dict)
                and "error" in state.code_execution_result
            ):
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages
                        + [
                            f"Code execution returned error: {state.code_execution_result['error']}"
                        ],
                        "data_validated": False,
                        "updated_at": datetime.now(),
                    },
                )

            # Get sample data appropriate for widget type
            sample_data = self._get_widget_sample(state.widget_config, state.widget_type)

            # Get structure analysis appropriate for widget type
            data_schema = self._analyze_widget_structure(state.widget_config, state.widget_type)

            # Fetch and compile validation prompt from Langfuse with dynamic variables
            try:
                # Extract sampling information for variables
                sampling_info = sample_data.get("_sampling_info", {})
                
                # Prepare runtime variables for Langfuse prompt compilation
                prompt_variables = {
                    "task_instructions": state.task_instructions,
                    "user_prompt": state.user_prompt,
                    "widget_type": state.widget_type,
                    "operation": state.operation,
                    "widget_config": json.dumps(state.widget_config, indent=2, default=str),
                    "sample_data": json.dumps(sample_data, indent=2, default=str),
                    "data_schema": json.dumps(data_schema, indent=2),
                    "sample_data_original_count": sampling_info.get("original_count", "unknown"),
                    "sample_data_sampled_count": sampling_info.get("sampled_count", "unknown"),
                    "sample_data_is_sampled": sampling_info.get("is_sampled", "unknown"),
                    "widget_title": state.title,
                    "widget_description": state.description
                }
                
                logger.info("Fetching and compiling validation prompt from Langfuse...")
                
                # Compile the prompt with dynamic variables from Langfuse (REQUIRED)
                validation_prompt = compile_prompt(
                    "widget_agent_team/validate_data", 
                    prompt_variables,
                    label="latest"
                )
                
                # Validate compiled prompt (handle different formats)
                if not validation_prompt:
                    raise ValueError("Compiled validation prompt from Langfuse is empty or None")
                
                # Convert to string if needed and validate
                validation_prompt_str = str(validation_prompt)
                if not validation_prompt_str or len(validation_prompt_str.strip()) == 0:
                    raise ValueError("Compiled validation prompt from Langfuse is empty or invalid")
                
                logger.info(f"✅ Successfully compiled Langfuse validation prompt with {len(prompt_variables)} variables")
                
                # Use the compiled prompt
                validation_prompt = validation_prompt_str
                
            except Exception as e:
                error_msg = f"Failed to fetch/compile validation prompt from Langfuse: {str(e)}"
                logger.error(error_msg)
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "data_validated": False,
                        "updated_at": datetime.now(),
                    },
                )

            # Create validation LLM with Langfuse configuration
            validation_llm_params = {
                "model": self.model,
                "temperature": self.temperature
            }
            
            # Add reasoning_effort if provided (for reasoning models like o1, o3, o4-mini)
            if self.reasoning_effort:
                validation_llm_params["reasoning_effort"] = self.reasoning_effort
                
            validation_llm = ChatOpenAI(**validation_llm_params).with_structured_output(DataValidationResult)

            # Get validation result from LLM
            validation_result = validation_llm.invoke(validation_prompt)

            # Create validation message
            validation_message = f"Confidence: {validation_result.confidence_level}% - {validation_result.explanation}"

            # Determine if data should be considered valid (high confidence threshold)
            data_is_valid = (
                validation_result.is_valid and validation_result.confidence_level >= 80
            )

            logger.info(f"🔍 VALIDATION RESULT: is_valid={validation_result.is_valid}, confidence={validation_result.confidence_level}%, data_is_valid={data_is_valid}")

            if data_is_valid:
                # Validation passed with high confidence - execute database operations immediately
                return Command(
                    goto="db_operations",  # Route to database operations node
                    update={
                        "data_validated": True,
                        "data": state.widget_config,  # Use unified widget_config
                        "updated_at": datetime.now(),
                        "messages": [
                            ToolMessage(
                                content=f"✅ Data validation successful! {validation_message}. Proceeding to database operations.",
                                tool_call_id="validation_complete",
                            )
                        ],
                    },
                )
            else:
                # Build detailed error message for retry
                error_details = []
                if validation_result.missing_requirements:
                    error_details.append(
                        f"Missing requirements: {', '.join(validation_result.missing_requirements)}"
                    )
                if validation_result.data_quality_issues:
                    error_details.append(
                        f"Data quality issues: {', '.join(validation_result.data_quality_issues)}"
                    )

                detailed_error = f"Validation failed (Confidence: {validation_result.confidence_level}%). {validation_result.explanation}"
                if error_details:
                    detailed_error += f" Issues: {'; '.join(error_details)}"

                return Command(
                    goto="widget_supervisor",
                    update={
                        "data_validated": False,
                        "error_messages": state.error_messages + [detailed_error],
                        "updated_at": datetime.now(),
                        "messages": [
                            ToolMessage(
                                content=f"❌ {validation_message}",
                                tool_call_id="validation_failed",
                            )
                        ],
                    },
                )

        except Exception as e:
            return Command(
                goto="widget_supervisor",
                update={
                    "error_messages": state.error_messages
                    + [f"Validation error: {str(e)}"],
                    "data_validated": False,
                    "updated_at": datetime.now(),
                },
            )

    def _get_widget_sample(self, widget_config, widget_type):
        """
        Extract sample data from widget_config for validation, handling both chart and text widgets.

        Args:
            widget_config: Dictionary containing widget configuration
            widget_type: Type of widget (chart types or 'text')

        Returns:
            Dictionary: Widget configuration with sampled data and metadata
        """
        if not isinstance(widget_config, dict):
            return widget_config

        # Create a copy of the full configuration
        sampled_result = widget_config.copy()

        if widget_type == "text":
            # For text widgets, sample the content
            if "content" in widget_config:
                content = widget_config["content"]
                if isinstance(content, str):
                    # Sample first 1000 characters of text content
                    original_length = len(content)
                    sampled_content = content[:1000]
                    
                    sampled_result["content"] = sampled_content
                    sampled_result["_sampling_info"] = {
                        "original_count": original_length,
                        "sampled_count": len(sampled_content),
                        "is_sampled": original_length > 1000,
                        "content_type": "text_content"
                    }
                else:
                    sampled_result["_sampling_info"] = {
                        "original_count": 1,
                        "sampled_count": 1,
                        "is_sampled": False,
                        "content_type": "non_text_content"
                    }
            else:
                sampled_result["_sampling_info"] = {
                    "original_count": 0,
                    "sampled_count": 0,
                    "is_sampled": False,
                    "warning": "Content field is missing for text widget"
                }
        else:
            # For chart widgets, sample the data array
            if "data" in widget_config and isinstance(widget_config["data"], list):
                original_data = widget_config["data"]
                sampled_data = original_data[:10]

                sampled_result["data"] = sampled_data
                sampled_result["_sampling_info"] = {
                    "original_count": len(original_data),
                    "sampled_count": len(sampled_data),
                    "is_sampled": len(original_data) > 10,
                    "content_type": "chart_data"
                }
            else:
                sampled_result["_sampling_info"] = {
                    "original_count": 0,
                    "sampled_count": 0,
                    "is_sampled": False,
                    "warning": "Data field is not a list or is missing for chart widget",
                    "content_type": "chart_data"
                }

        return sampled_result

    def _analyze_widget_structure(self, widget_config, widget_type):
        """Analyze the structure and types of widget configuration data."""
        if not isinstance(widget_config, dict):
            return {
                "type": type(widget_config).__name__,
                "value": str(widget_config)[:100],
            }

        analysis = {
            "widget_type": widget_type,
            "schema_fields": {},
            "analysis_type": "text_widget" if widget_type == "text" else "chart_widget",
        }

        if widget_type == "text":
            # Analyze text widget structure
            text_fields = ["content", "title", "description"]
            for field in text_fields:
                if field in widget_config:
                    field_value = widget_config[field]
                    analysis["schema_fields"][field] = {
                        "type": type(field_value).__name__,
                        "present": True,
                    }

                    # Special analysis for content field
                    if field == "content" and isinstance(field_value, str):
                        analysis["schema_fields"][field].update({
                            "content_length": len(field_value),
                            "has_html": "<" in field_value and ">" in field_value,
                            "preview": field_value[:200] + "..." if len(field_value) > 200 else field_value
                        })
                else:
                    analysis["schema_fields"][field] = {"type": "missing", "present": False}
        else:
            # Analyze chart widget structure (ChartConfigSchema)
            chart_fields = ["chartType", "title", "description", "data", "chartConfig", "xAxisConfig"]
            for field in chart_fields:
                if field in widget_config:
                    field_value = widget_config[field]
                    analysis["schema_fields"][field] = {
                        "type": type(field_value).__name__,
                        "present": True,
                    }

                    # Special analysis for data field
                    if field == "data" and isinstance(field_value, list):
                        analysis["schema_fields"][field].update({
                            "length": len(field_value),
                            "sample_keys": list(field_value[0].keys())
                            if field_value and isinstance(field_value[0], dict)
                            else None,
                            "item_type": type(field_value[0]).__name__
                            if field_value
                            else "unknown",
                        })
                    # Special analysis for chartConfig
                    elif field == "chartConfig" and isinstance(field_value, dict):
                        analysis["schema_fields"][field].update({
                            "config_keys": list(field_value.keys()),
                            "config_count": len(field_value),
                        })
                    # Special analysis for xAxisConfig
                    elif field == "xAxisConfig" and isinstance(field_value, dict):
                        analysis["schema_fields"][field].update(
                            {"dataKey": field_value.get("dataKey", "not_specified")}
                        )
                else:
                    analysis["schema_fields"][field] = {"type": "missing", "present": False}

        # Add sampling info if present
        if "_sampling_info" in widget_config:
            analysis["sampling_info"] = widget_config["_sampling_info"]

        return analysis

