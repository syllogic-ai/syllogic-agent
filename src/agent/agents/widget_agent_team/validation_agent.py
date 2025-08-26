"""Data validation agent for widget processing."""

import json
from datetime import datetime
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.models import WidgetAgentState

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
            import logging
            logger = logging.getLogger(__name__)
            
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
            import logging
            logging.getLogger(__name__).error(error_msg)
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

            # Get first 10 rows/items of generated data for validation
            sample_data = self._get_data_sample(state.widget_config)

            # Get data schema information
            data_schema = self._analyze_data_structure(state.widget_config)

            # Fetch and compile validation prompt from Langfuse with dynamic variables
            try:
                import logging
                logger = logging.getLogger(__name__)
                
                # Extract sampling information for variables
                sampling_info = sample_data.get("_sampling_info", {})
                
                # Prepare runtime variables for Langfuse prompt compilation
                prompt_variables = {
                    "task_instructions": state.task_instructions,
                    "user_prompt": state.user_prompt,
                    "widget_type": state.widget_type,
                    "operation": state.operation,
                    "sample_data": json.dumps(sample_data, indent=2, default=str),
                    "data_schema": json.dumps(data_schema, indent=2),
                    "sample_data_original_count": sampling_info.get("original_count", "unknown"),
                    "sample_data_sampled_count": sampling_info.get("sampled_count", "unknown"),
                    "sample_data_is_sampled": sampling_info.get("is_sampled", "unknown")
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
                import logging
                logging.getLogger(__name__).error(error_msg)
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

            if data_is_valid:
                # Continue to DB operations after validation passes
                return Command(
                    goto="widget_supervisor",
                    update={
                        "data_validated": True,
                        "data": state.widget_config,  # Use unified widget_config
                        "updated_at": datetime.now(),
                        "messages": [
                            ToolMessage(
                                content=f"✅ Data validation successful! {validation_message}",
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

    def _get_data_sample(self, chart_config_data):
        """
        Extract first 10 records from ChartConfigSchema data for validation.

        Args:
            chart_config_data: Dictionary following ChartConfigSchema structure

        Returns:
            Dictionary: Complete ChartConfigSchema structure with sampled data (first 10 records)
        """
        if not isinstance(chart_config_data, dict):
            # Fallback for non-dict data (shouldn't happen with validated schema)
            return chart_config_data

        # Create a copy of the full schema
        sampled_result = chart_config_data.copy()

        # Extract and sample the data array
        if "data" in chart_config_data and isinstance(chart_config_data["data"], list):
            # Get first 10 records from the data array
            original_data = chart_config_data["data"]
            sampled_data = original_data[:10]

            # Update the result with sampled data
            sampled_result["data"] = sampled_data

            # Add metadata about sampling
            sampled_result["_sampling_info"] = {
                "original_count": len(original_data),
                "sampled_count": len(sampled_data),
                "is_sampled": len(original_data) > 10,
            }
        else:
            # If data is not a list or doesn't exist, return as-is with warning
            sampled_result["_sampling_info"] = {
                "original_count": 0,
                "sampled_count": 0,
                "is_sampled": False,
                "warning": "Data field is not a list or is missing",
            }

        return sampled_result

    def _analyze_data_structure(self, chart_config_data):
        """Analyze the structure and types of ChartConfigSchema data."""
        if not isinstance(chart_config_data, dict):
            return {
                "type": type(chart_config_data).__name__,
                "value": str(chart_config_data)[:100],
            }

        analysis = {
            "type": "ChartConfigSchema",
            "schema_fields": {},
            "data_analysis": {},
        }

        # Analyze each schema field
        schema_fields = [
            "chartType",
            "title",
            "description",
            "data",
            "chartConfig",
            "xAxisConfig",
        ]
        for field in schema_fields:
            if field in chart_config_data:
                field_value = chart_config_data[field]
                analysis["schema_fields"][field] = {
                    "type": type(field_value).__name__,
                    "present": True,
                }

                # Special analysis for data field
                if field == "data" and isinstance(field_value, list):
                    analysis["schema_fields"][field].update(
                        {
                            "length": len(field_value),
                            "sample_keys": list(field_value[0].keys())
                            if field_value and isinstance(field_value[0], dict)
                            else None,
                            "item_type": type(field_value[0]).__name__
                            if field_value
                            else "unknown",
                        }
                    )
                # Special analysis for chartConfig
                elif field == "chartConfig" and isinstance(field_value, dict):
                    analysis["schema_fields"][field].update(
                        {
                            "config_keys": list(field_value.keys()),
                            "config_count": len(field_value),
                        }
                    )
                # Special analysis for xAxisConfig
                elif field == "xAxisConfig" and isinstance(field_value, dict):
                    analysis["schema_fields"][field].update(
                        {"dataKey": field_value.get("dataKey", "not_specified")}
                    )
            else:
                analysis["schema_fields"][field] = {"type": "missing", "present": False}

        # Add sampling info if present
        if "_sampling_info" in chart_config_data:
            analysis["sampling_info"] = chart_config_data["_sampling_info"]

        return analysis