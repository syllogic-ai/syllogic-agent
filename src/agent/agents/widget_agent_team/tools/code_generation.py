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

        # Create a code generation agent
        code_gen_llm = ChatOpenAI(model="gpt-4o-mini")

        code_generation_prompt = f"""
You are a Python data analysis expert specializing in pandas and data manipulation. Given a dataset schema, sample data, and a user's request, generate valid Python code to analyze and manipulate the data for the specified widget type and operation.

USER REQUEST: {user_request}
WIDGET TYPE: {widget_type}
OPERATION: {operation}

FILE SCHEMAS:
{json.dumps(schemas_info, indent=2) if schemas_info else "No schemas available"}

SAMPLE DATA INFO:
{json.dumps(samples_info, indent=2) if samples_info else "No sample data available"}

DATA CONTEXT:
- A pandas DataFrame named 'df' is available in the execution environment
- The DataFrame is already loaded with your file data
- You can inspect the DataFrame with: print(df.shape), print(df.columns), print(df.head())
- All necessary imports (pandas as pd, numpy as np, json) are already done

IMPORTANT INSTRUCTIONS:
- Adhere strictly to column names: Use the exact column names and casing as provided in the 'Schema' section. If a column name includes spaces or special characters, access it using bracket notation (e.g., df["Column Name with Spaces"], df["caseSensitiveName"]).
- Don't explain your reasoning - just return the Python code.
- Think hard about which columns or rows you need to use for the operation.
- Use proper pandas syntax and best practices.
- Always assume the DataFrame is named 'df' and is already loaded.
- Start your code by inspecting the data if needed: df.info(), df.head(), df.columns
- For aggregations, use appropriate pandas groupby operations.
- For visualizations, prepare data in a format suitable for plotting libraries.
- AVOID operations that modify the original dataset permanently (like df.drop(inplace=True)).
- Don't include import statements - assume pandas is imported as 'pd' and numpy as 'np'.
- The DataFrame variable name is ALWAYS "df"
- Ensure all column references are properly handled for spaces and special characters
- Use pandas-compatible date functions (pd.to_datetime, dt accessor methods)
- Include proper null handling using .fillna(), .dropna(), or .isna() methods
- For percentage calculations, use proper formula: ((new - old) / old) * 100
- Check that all referenced columns exist in the schema before using them
- Use explicit type conversion when needed (.astype(), pd.to_numeric(), pd.to_datetime())
- Ensure proper handling of conditional logic using np.where(), df.loc[], or boolean indexing
- Use .copy() when creating derived DataFrames to avoid SettingWithCopyWarning
- For temporal analysis, use pandas datetime methods consistently
- Handle missing or invalid data gracefully
- Use vectorized operations instead of loops where possible
- For complex filtering, use boolean indexing or .query() method
- Date columns should be converted to datetime if needed: pd.to_datetime(df['date_column'])
- Only work with the available column types: object (TEXT), int64 (INTEGER), float64 (FLOAT), bool (BOOLEAN), datetime64 (DATE/TIMESTAMP)
- Don't apply unnecessary formatting to numerical columns, except for display purposes
- Use appropriate pandas methods for statistical operations

PANDAS DATE/TIME REQUIREMENTS:
- Always use pd.to_datetime() to convert date strings to datetime objects
- For current date, use pd.Timestamp.now() or pd.Timestamp.today()
- For date filtering, use: df[df['date_column'].dt.date == pd.Timestamp.today().date()]
- For date/time extraction, use dt accessor: df['date_column'].dt.year, df['date_column'].dt.month
- For date comparisons, ensure both sides are datetime objects
- When parsing custom date formats, use: pd.to_datetime(df['date_column'], format='%d/%m/%Y')
- For date truncation by month: df['date_column'].dt.to_period('M').dt.start_time (NOT dt.floor('M'))
- For date truncation by day: df['date_column'].dt.floor('D')
- Use dt.strftime() for date formatting: df['date_column'].dt.strftime('%Y-%m-%d')
- Handle timezone-aware dates using dt.tz_localize() or dt.tz_convert()
- For date arithmetic, use pd.Timedelta: df['date_column'] + pd.Timedelta(days=1)
- CRITICAL: Never use dt.floor('M') - use dt.to_period('M').dt.start_time for monthly grouping

REQUIRED OUTPUT SCHEMA:
Your generated Python code MUST produce a result that follows this exact schema structure:

{chart_config_schema}

CRITICAL REQUIREMENTS:
- The final line MUST assign the result to a variable named 'result'
- The result MUST be a dictionary that matches the ChartConfigSchema structure exactly
- ALWAYS end with: result = {{"chartType": "{widget_type}", "title": "...", "description": "...", "data": [...], "chartConfig": {{}}, "xAxisConfig": {{}}}}
- DO NOT end with print() statements - use result = instead

SCHEMA COMPLIANCE:
- chartType: Must be one of: "line", "bar", "pie", "area", "radial", "kpi", "table"
- title: String describing the chart/widget
- description: String explaining what the chart shows
- data: Array of objects containing the actual data points
- chartConfig: Dictionary where keys are data field names and values contain label and color
- xAxisConfig: Object with dataKey specifying which field to use for X-axis

EXAMPLE RESULT STRUCTURE:
```python
result = {{
    "chartType": "{widget_type}",
    "title": "Your Chart Title",
    "description": "Description of what this chart shows",
    "data": [
        {{"category": "Jan", "value": 100}},
        {{"category": "Feb", "value": 150}},
        # ... more data points
    ],
    "chartConfig": {{
        "value": {{
            "label": "Value Label",
            "color": "var(--chart-1)"
        }}
        # Add more chart config items for each data field
    }},
    "xAxisConfig": {{
        "dataKey": "category"
    }}
}}
```

IMPORTANT:
- Return executable Python code that processes the DataFrame 'df'
- Code should be optimized for the specified WIDGET TYPE ({widget_type}) and OPERATION ({operation})
- Ensure the code is ready to execute without additional setup
- OPTIONAL: Start with basic data inspection if needed: print(df.shape), print(df.columns)
- If the DataFrame appears empty or missing expected columns, include error handling
- The result MUST strictly follow the ChartConfigSchema structure above

USER REQUEST: {user_request}

AVAILABLE DATA SUMMARY:
- File count: {len(schemas_info)}
- Total rows available: {sum(schema.get("total_rows", 0) for schema in schemas_info)}
- Column information: {", ".join([f"{schema.get('file_id', 'unknown')}: {len(schema.get('columns', []))} columns" for schema in schemas_info])}

REMEMBER: The DataFrame 'df' is already loaded with your data. You can immediately start working with it.

"""

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