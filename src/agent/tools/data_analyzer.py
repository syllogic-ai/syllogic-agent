"""Data analysis tool that uses LLM to compare current data with user query."""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..models import TopLevelState

logger = logging.getLogger(__name__)


@tool
async def analyze_data_sufficiency(
    state: TopLevelState, llm: ChatOpenAI = None
) -> Dict[str, Any]:
    """Use LLM to analyze if available data is sufficient for the user's request.

    Args:
        state: Current chart generation state
        llm: Optional LLM instance (will create one if not provided)

    Returns:
        Dict with analysis results:
        {
            "sufficient": bool,
            "missing_data": List[str],
            "recommendations": str,
            "confidence": float
        }
    """
    try:
        logger.info("Starting data sufficiency analysis")

        if not llm:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Prepare data summary for LLM analysis
        data_summary = _prepare_data_summary(state)

        # Create analysis prompt
        system_prompt = """
        You are a data analysis expert. Your task is to analyze if the available data is sufficient to fulfill the user's request.
        
        You need to determine:
        1. Whether the available data can fulfill the user's request
        2. What specific data might be missing
        3. Recommendations for proceeding
        
        Respond in JSON format with:
        {
            "sufficient": boolean (true if data is adequate),
            "missing_data": array of strings (specific data requirements that are missing),
            "recommendations": string (actionable recommendations),
            "confidence": float (0.0 to 1.0, how confident you are in this analysis)
        }
        """

        user_prompt = f"""
        USER REQUEST: {state.user_prompt}
        
        AVAILABLE DATA:
        {data_summary}
        
        Analyze if this data is sufficient to fulfill the user's request.
        """

        # Get LLM analysis
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)

        # Parse response (assuming it's valid JSON)
        import json

        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            analysis = {
                "sufficient": False,
                "missing_data": ["Unable to parse LLM response"],
                "recommendations": "Retry analysis with clearer prompt",
                "confidence": 0.1,
            }

        logger.info(
            f"Data sufficiency analysis completed. Sufficient: {analysis.get('sufficient', False)}"
        )
        return analysis

    except Exception as e:
        error_msg = f"Error during data sufficiency analysis: {str(e)}"
        logger.error(error_msg)
        return {
            "sufficient": False,
            "missing_data": [error_msg],
            "recommendations": "Review data and try again",
            "confidence": 0.0,
        }


@tool
def _prepare_data_summary(state: TopLevelState) -> str:
    """Prepare a readable summary of available data for LLM analysis.

    Args:
        state: Current chart generation state

    Returns:
        String summary of available data
    """
    try:
        if not state.available_data_schemas and not state.available_sample_data:
            return "No data available."

        summary_parts = []

        # Add schemas information
        if state.available_data_schemas:
            summary_parts.append(
                f"DATA SCHEMAS ({len(state.available_data_schemas)} files):"
            )

            for file_id, schema in state.available_data_schemas.items():
                summary_parts.append(f"\nFile: {file_id}")
                summary_parts.append(f"  Total rows: {schema.total_rows}")
                summary_parts.append(f"  Total columns: {schema.total_columns}")
                summary_parts.append("  Columns:")

                for col in schema.columns[:10]:  # Limit to first 10 columns
                    sample_str = (
                        f" (samples: {col.sample_values[:3]})"
                        if col.sample_values
                        else ""
                    )
                    summary_parts.append(f"    - {col.name} ({col.type}){sample_str}")

                if len(schema.columns) > 10:
                    summary_parts.append(
                        f"    ... and {len(schema.columns) - 10} more columns"
                    )

        # Add sample data information
        if state.available_sample_data:
            summary_parts.append(
                f"\nSAMPLE DATA ({len(state.available_sample_data)} files):"
            )

            for file_id, sample in state.available_sample_data.items():
                summary_parts.append(f"\nFile: {file_id}")
                summary_parts.append(f"  Headers: {', '.join(sample.headers[:10])}")
                if len(sample.headers) > 10:
                    summary_parts.append(
                        f"  ... and {len(sample.headers) - 10} more columns"
                    )

                summary_parts.append(
                    f"  Sample rows ({sample.sample_rows_returned} of {sample.total_rows_in_file}):"
                )
                for i, row in enumerate(sample.rows[:3]):  # Show first 3 rows
                    row_str = ", ".join(
                        [str(val)[:20] for val in row[:5]]
                    )  # First 5 columns, truncate long values
                    if len(row) > 5:
                        row_str += f", ... (+{len(row) - 5} more)"
                    summary_parts.append(f"    {i + 1}: {row_str}")

        return "\n".join(summary_parts)

    except Exception as e:
        logger.error(f"Error preparing data summary: {str(e)}")
        return f"Error preparing data summary: {str(e)}"
