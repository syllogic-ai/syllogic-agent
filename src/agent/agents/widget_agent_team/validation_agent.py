"""Data validation agent for widget processing."""

import json
from datetime import datetime
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.models import WidgetAgentState


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
        """Initialize validation agent with LLM."""
        self.llm_model = llm_model

    def validate_data(self, state: WidgetAgentState) -> Command:
        """LLM-based validation that analyzes generated data against user requirements."""

        try:
            # Check if we have execution result
            if state.code_execution_result is None:
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages
                        + ["No execution result to validate"],
                        "data_validated": False,
                        "updated_at": datetime.now(),
                    },
                )

            # Check for execution errors first
            if (
                isinstance(state.code_execution_result, dict)
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
            sample_data = self._get_data_sample(state.code_execution_result)

            # Get data schema information
            data_schema = self._analyze_data_structure(state.code_execution_result)

            # Create validation prompt for LLM
            validation_llm = init_chat_model(
                self.llm_model
            ).with_structured_output(DataValidationResult)

            validation_prompt = f"""
You are a data validation expert specializing in sample-based quality assessment. Your task is to evaluate whether the generated data sample is "on the right track" and aligns with user requirements.

⚠️ CRITICAL CONTEXT: You are analyzing only a SAMPLE (first 10 records) of what may be a much larger dataset. Do NOT penalize for missing data points that aren't in the sample - focus on whether the sample demonstrates correct alignment.

TASK DESCRIPTION: {state.task_instructions}
USER PROMPT: {state.user_prompt}
WIDGET TYPE: {state.widget_type}
OPERATION: {state.operation}

GENERATED CHART CONFIG SAMPLE (first 10 records only):
{json.dumps(sample_data, indent=2, default=str)}

CHART CONFIG ANALYSIS:
{json.dumps(data_schema, indent=2)}

SAMPLE CONTEXT:
Original dataset: {sample_data.get("_sampling_info", {}).get("original_count", "unknown")} records
Sample shown: {sample_data.get("_sampling_info", {}).get("sampled_count", "unknown")} records
Is sampled: {sample_data.get("_sampling_info", {}).get("is_sampled", "unknown")}

EVALUATION CRITERIA (Sample-Based):
✅ Structure Validation: Is the ChartConfigSchema properly formed with all required fields?
✅ Data Alignment: Do the sample records match the user's request (correct field names, logical values)?
✅ Chart Configuration: Are chartConfig mappings sensible for the data fields present?
✅ Axis Setup: Is xAxisConfig.dataKey pointing to a valid field that exists in the sample?
✅ Metadata Quality: Are title/description relevant and descriptive?
✅ Data Logic: Does the sample data follow patterns that make sense for the request?

VALIDATION APPROACH:
- HIGH CONFIDENCE (85+): Sample clearly demonstrates correct approach, structure, and alignment
- MEDIUM CONFIDENCE (70-84): Sample shows mostly correct approach with minor issues
- LOW CONFIDENCE (<70): Sample has structural problems or clear misalignment

KEY PRINCIPLE: If the sample is well-structured and shows the right data patterns, assume the full dataset follows the same logic. Do NOT reduce confidence just because you're seeing a subset.

Provide:
- is_valid: Boolean - true if sample demonstrates correct direction and structure
- confidence_level: 0-100 score based on sample quality and alignment (not completeness)
- explanation: Focus on whether the sample indicates success for the full dataset
- missing_requirements: Only list structural/configuration issues visible in the sample
- data_quality_issues: Only list problems evident in the actual sample data shown

Remember: You're validating the APPROACH and QUALITY, not the completeness!
"""

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
                        "data": state.code_execution_result,
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