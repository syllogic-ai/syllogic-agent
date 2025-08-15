"""Worker nodes for widget data processing pipeline using create_react_agent."""

import json
import os

# Import e2b sandbox creation function
import sys
from datetime import datetime
from typing import Annotated, Any, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.models import (
    ChartConfigSchema,
    ColumnInfo,
    FileSampleData,
    FileSchema,
    WidgetAgentState,
)

# Add src directory to Python path for config import
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "..", "..")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from typing import Annotated, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command

from config import create_e2b_sandbox

# E2B sandbox will be created fresh for each execution to avoid state contamination

# No global variables - use Command objects for immediate state updates


def get_chart_config_schema_string():
    """
    Programmatically extract the ChartConfigSchema as a formatted string for code generation prompts.

    Returns:
        str: A formatted string representation of the ChartConfigSchema that can be used in prompts.
    """
    try:
        # Get the JSON schema for ChartConfigSchema
        schema = ChartConfigSchema.model_json_schema()

        # Extract properties and their descriptions
        properties = schema.get("properties", {})

        # Build a formatted string representation
        schema_lines = []
        schema_lines.append("ChartConfigSchema Structure:")
        schema_lines.append("```json")
        schema_lines.append("{")

        # Add each property with its type and description
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get(
                "type", prop_info.get("anyOf", [{}])[0].get("type", "unknown")
            )
            description = prop_info.get("description", "No description")

            # Handle special cases for complex types
            if prop_name == "chartType":
                enum_values = prop_info.get("enum", [])
                schema_lines.append(
                    f'  "{prop_name}": "{" | ".join(enum_values)}" // {description}'
                )
            elif prop_name == "chartConfig":
                schema_lines.append(f'  "{prop_name}": {{')
                schema_lines.append(f'    "<item_name>": {{')
                schema_lines.append(
                    f'      "label": "string", // Display label for the chart item'
                )
                schema_lines.append(
                    f'      "color": "string"  // Color value (e.g., "var(--chart-1)" or hex)'
                )
                schema_lines.append(f"    }}")
                schema_lines.append(f"  }}, // {description}")
            elif prop_name == "xAxisConfig":
                schema_lines.append(f'  "{prop_name}": {{')
                schema_lines.append(
                    f'    "dataKey": "string" // The data key to use for the X-axis'
                )
                schema_lines.append(f"  }}, // {description}")
            elif prop_name == "data":
                schema_lines.append(f'  "{prop_name}": [ // {description}')
                schema_lines.append(
                    f'    {{ "key": "value", ... }}, // Array of data objects'
                )
                schema_lines.append(f"    ...")
                schema_lines.append(f"  ],")
            else:
                schema_lines.append(f'  "{prop_name}": "{prop_type}" // {description}')

        schema_lines.append("}")
        schema_lines.append("```")

        # Add the example from the model's schema
        example = schema.get("example")
        if example and isinstance(example, dict):
            schema_lines.append("\nExample:")
            schema_lines.append("```json")
            schema_lines.append(json.dumps(example, indent=2))
            schema_lines.append("```")

        return "\n".join(schema_lines)

    except Exception as e:
        # Fallback to a hardcoded schema if extraction fails
        return """ChartConfigSchema Structure:
```json
{
  "chartType": "line | bar | pie | area | radial | kpi | table", // Type of chart
  "title": "string", // Title of the chart
  "description": "string", // Description of the chart
  "data": [ // Data for the chart
    { "key": "value", ... }, // Array of data objects
    ...
  ],
  "chartConfig": {
    "<item_name>": {
      "label": "string", // Display label for the chart item
      "color": "string"  // Color value (e.g., "var(--chart-1)" or hex)
    }
  }, // Dictionary of chart items
  "xAxisConfig": {
    "dataKey": "string" // The data key to use for the X-axis
  } // X-axis configuration
}
```"""


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


@tool
def fetch_data_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Fetches data from file IDs in state and updates state with fetched data."""
    try:
        # Import file fetching functions from actions
        try:
            from actions.dashboard import (
                get_data_from_file,
                get_sample_from_file,
                get_schema_from_file,
            )
        except ImportError as import_error:
            # Try alternative import paths
            try:
                import os
                import sys

                # Add src to path if not already there
                current_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.join(current_dir, "..", "..", "..")  # Go up to src/
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)

                from actions.dashboard import (
                    get_data_from_file,
                    get_sample_from_file,
                    get_schema_from_file,
                )
            except ImportError as secondary_error:
                return f"Error: Actions module not available for file fetching. Primary: {str(import_error)}, Secondary: {str(secondary_error)}, Current path: {sys.path[:5]}"

        # Extract file_ids from state
        file_ids = state.file_ids
        if not file_ids:
            return Command(
                update={
                    "error_messages": state.error_messages
                    + ["No file IDs provided for data fetching"],
                    "messages": [
                        ToolMessage(
                            content="Error: No file IDs provided for data fetching",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        raw_data = {}
        schemas_info = []
        samples_info = []

        for file_id in file_ids:
            try:
                # Fetch raw data for processing
                file_data = get_data_from_file(file_id)
                raw_data[file_id] = file_data

                # Always fetch schema and sample data
                schema_data = get_schema_from_file(file_id)
                schemas_info.append(
                    {
                        "file_id": file_id,
                        "columns": len(schema_data.get("columns", [])),
                        "total_rows": schema_data.get("total_rows", 0),
                        "column_names": [
                            col["name"] for col in schema_data.get("columns", [])
                        ],
                    }
                )

                sample_data = get_sample_from_file(file_id)
                samples_info.append(
                    {
                        "file_id": file_id,
                        "headers": sample_data.get("headers", []),
                        "sample_rows": sample_data.get("sample_rows_returned", 0),
                    }
                )

            except Exception as file_error:
                return f"Failed to fetch data for file {file_id}: {str(file_error)}"

        # Convert raw pandas DataFrames to JSON for state storage
        raw_data_json = {}
        for file_id, df in raw_data.items():
            # Convert DataFrame to dict with records orientation
            raw_data_json[file_id] = df.to_dict("records")

        # Create FileSchema objects
        file_schemas_objects = [
            FileSchema(
                file_id=info["file_id"],
                columns=[
                    ColumnInfo(name=col, type="unknown", null_count=0, unique_count=0)
                    for col in info["column_names"]
                ],
                total_rows=info["total_rows"],
                total_columns=len(info["column_names"]),
            )
            for info in schemas_info
        ]

        # Create FileSampleData objects
        file_sample_objects = [
            FileSampleData(
                file_id=info["file_id"],
                headers=info["headers"],
                rows=[],  # Could populate from sample data if needed
                total_rows_in_file=schemas_info[i]["total_rows"]
                if i < len(schemas_info)
                else 0,
                sample_rows_returned=info["sample_rows"],
            )
            for i, info in enumerate(samples_info)
        ]

        success_message = (
            f"Successfully fetched data from {len(file_ids)} files:\n"
            + "\n".join(
                [
                    f"File {info['file_id']}: {info['total_rows']} rows, columns: {', '.join(info['column_names'][:5])}{'...' if len(info['column_names']) > 5 else ''}"
                    for info in schemas_info
                ]
            )
        )

        # Return Command to immediately update state
        return Command(
            update={
                "raw_file_data": raw_data_json,
                "file_schemas": file_schemas_objects,
                "file_sample_data": file_sample_objects,
                "messages": [
                    ToolMessage(content=success_message, tool_call_id=tool_call_id)
                ],
            }
        )

    except Exception as e:
        error_msg = f"Data fetch error: {str(e)}"
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


@tool
def e2b_sandbox_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Executes Python code using E2B sandbox with state data context."""
    try:
        # Get the generated code from state
        code = state.generated_code
        if not code:
            return Command(
                update={
                    "error_messages": state.error_messages
                    + ["No generated code to execute"],
                    "messages": [
                        ToolMessage(
                            content="Error: No generated code to execute",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        # Get raw data from state
        raw_data = state.raw_file_data or {}

        # Debug: Check if we have raw data
        if not raw_data:
            error_msg = "No raw file data available in state. The fetch_data_tool may not have run successfully."
            return Command(
                update={
                    "code_execution_result": {"error": "No raw data available"},
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [
                        ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                    ],
                }
            )

        # Create a fresh E2B sandbox instance for each execution
        try:
            sandbox = create_e2b_sandbox()
            print(f"E2B sandbox created successfully: {type(sandbox)}")
        except Exception as sandbox_error:
            error_msg = f"Failed to create E2B sandbox: {str(sandbox_error)}"
            print(f"E2B Sandbox Creation Error: {error_msg}")
            return Command(
                update={
                    "code_execution_result": {"error": error_msg},
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [
                        ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                    ],
                }
            )

        with sandbox:
            # Prepare the execution context with raw_data and reconstruct DataFrames
            # Avoid f-string nesting issues by pre-serializing the data
            raw_data_json = json.dumps(raw_data, default=str)

            context_setup = f"""
import json
import pandas as pd
import numpy as np

# Load the raw data
raw_data = {raw_data_json}

# Reconstruct DataFrames from raw_data JSON
print("Available raw data keys:", list(raw_data.keys()) if raw_data else "No data")

if raw_data:
    # For single file, create df variable
    if len(raw_data) == 1:
        file_ids = list(raw_data.keys())
        first_file_id = file_ids[0]
        df = pd.DataFrame(raw_data[first_file_id])
        print("Created df from file", first_file_id, "with shape:", df.shape)
        print("DataFrame columns:", list(df.columns))
    else:
        # For multiple files, create individual DataFrames
        for current_file_id, data in raw_data.items():
            safe_name = current_file_id.replace("-", "_")
            df_name = "df_" + safe_name
            globals()[df_name] = pd.DataFrame(data)
            print("Created", df_name, "with shape:", globals()[df_name].shape)
        
        # Also create a combined df if possible
        try:
            all_data = []
            for data in raw_data.values():
                all_data.extend(data)
            df = pd.DataFrame(all_data)
            print("Created combined df with shape:", df.shape)
        except Exception as e:
            print("Could not combine data:", str(e))
            # If combination fails, use the first DataFrame as df
            file_ids = list(raw_data.keys())
            first_file_id = file_ids[0]
            df = pd.DataFrame(raw_data[first_file_id])
            print("Using first file df with shape:", df.shape)
else:
    df = pd.DataFrame()  # Create empty DataFrame if no data
    print("Warning: No raw data available, created empty DataFrame")
"""

            # Execute context setup first
            setup_execution = sandbox.run_code(context_setup)
            if setup_execution.error:
                error_msg = f"Context setup failed: {setup_execution.error}"
                return Command(
                    update={
                        "code_execution_result": {"error": setup_execution.error},
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [
                            ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                        ],
                    }
                )

            # Get output from logs.stdout since text is None in this E2B version
            setup_output = ""
            if setup_execution.logs and setup_execution.logs.stdout:
                setup_output = "".join(setup_execution.logs.stdout)
            print(
                f"Context setup output: {setup_output if setup_output else 'No output'}"
            )

            # Add a verification step to check if DataFrame was loaded properly
            verification_code = """
print("=== DATA VERIFICATION ===")
print(f"DataFrame 'df' exists: {'df' in locals() or 'df' in globals()}")
if 'df' in locals() or 'df' in globals():
    df_ref = df if 'df' in locals() else globals()['df']
    print(f"DataFrame shape: {df_ref.shape}")
    print(f"DataFrame columns: {list(df_ref.columns)}")
    print(f"First few rows:")
    print(df_ref.head(3))
else:
    print("ERROR: DataFrame 'df' not found!")
print("=========================")
"""
            verification_execution = sandbox.run_code(verification_code)
            if verification_execution.error:
                print(f"Verification warning: {verification_execution.error}")
            else:
                verification_output = ""
                if verification_execution.logs and verification_execution.logs.stdout:
                    verification_output = "".join(verification_execution.logs.stdout)
                print(
                    f"Verification output: {verification_output if verification_output else 'No output'}"
                )

            # Execute the main code and capture any execution errors
            main_execution = sandbox.run_code(code)
            if main_execution.error:
                error_msg = f"Python execution error: {main_execution.error}"
                return Command(
                    update={
                        "code_execution_result": {"error": main_execution.error},
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [
                            ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                        ],
                    }
                )

            # Try to extract 'result' variable from the sandbox session
            try:
                extract_result_code = """
import json
if 'result' in locals() or 'result' in globals():
    try:
        result_value = result if 'result' in locals() else globals()['result']
        print("RESULT_START")
        print(json.dumps(result_value, default=str))
        print("RESULT_END")
    except Exception as e:
        print("RESULT_START")
        print(json.dumps({"error": f"Failed to serialize result: {str(e)}"}, default=str))
        print("RESULT_END")
else:
    print("RESULT_START")
    print(json.dumps({"error": "No result variable found in the generated code. The code should end with 'result = final_output'"}, default=str))
    print("RESULT_END")
"""
                extract_execution = sandbox.run_code(extract_result_code)

                if extract_execution.error:
                    error_msg = f"Result extraction failed: {extract_execution.error}"
                    return Command(
                        update={
                            "code_execution_result": {
                                "error": extract_execution.error,
                                "raw_output": main_execution.text
                                if main_execution.text
                                else "No output captured",
                            },
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [
                                ToolMessage(
                                    content=error_msg, tool_call_id=tool_call_id
                                )
                            ],
                        }
                    )

                # Get output from logs.stdout since text is None in this E2B version
                output = ""
                if extract_execution.logs and extract_execution.logs.stdout:
                    output = "".join(extract_execution.logs.stdout)

                # Handle case where execution returns no output
                if not output:
                    # Get main execution output for debugging
                    main_output = ""
                    if main_execution.logs and main_execution.logs.stdout:
                        main_output = "".join(main_execution.logs.stdout)

                    error_msg = "E2B sandbox execution returned no output. This usually indicates an API connection issue or the code didn't execute properly."
                    return Command(
                        update={
                            "code_execution_result": {
                                "error": "No output from sandbox",
                                "raw_output": main_output
                                if main_output
                                else "No output captured",
                            },
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [
                                ToolMessage(
                                    content=error_msg, tool_call_id=tool_call_id
                                )
                            ],
                        }
                    )

                # Extract the JSON result from the output
                if "RESULT_START" in output and "RESULT_END" in output:
                    start_idx = output.find("RESULT_START") + len("RESULT_START")
                    end_idx = output.find("RESULT_END")
                    result_json = output[start_idx:end_idx].strip()
                    parsed_result = json.loads(result_json)

                    # Return Command with execution result
                    success_msg = f"Code executed successfully. Result: {json.dumps(parsed_result, indent=2)[:500]}{'...' if len(str(parsed_result)) > 500 else ''}"
                    return Command(
                        update={
                            "code_execution_result": parsed_result,
                            "messages": [
                                ToolMessage(
                                    content=success_msg, tool_call_id=tool_call_id
                                )
                            ],
                        }
                    )
                else:
                    # Get main execution output for debugging
                    main_output = ""
                    if main_execution.logs and main_execution.logs.stdout:
                        main_output = "".join(main_execution.logs.stdout)

                    # Return raw output with error context for retry
                    error_msg = f"Could not extract result from code execution. Raw output: {str(main_output)[:500]}{'...' if len(str(main_output)) > 500 else ''}"
                    return Command(
                        update={
                            "code_execution_result": {
                                "error": "No result variable found",
                                "raw_output": main_output,
                            },
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [
                                ToolMessage(
                                    content=error_msg, tool_call_id=tool_call_id
                                )
                            ],
                        }
                    )

            except Exception as extract_error:
                # If extraction fails, return error with context for potential retry
                # Get main execution output for debugging
                main_output = ""
                if main_execution.logs and main_execution.logs.stdout:
                    main_output = "".join(main_execution.logs.stdout)

                error_msg = f"Result extraction failed: {str(extract_error)}. This usually means the generated code doesn't end with 'result = final_output'"
                return Command(
                    update={
                        "code_execution_result": {
                            "error": str(extract_error),
                            "raw_output": main_output
                            if main_output
                            else "No output captured",
                        },
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [
                            ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                        ],
                    }
                )

    except Exception as e:
        error_msg = f"E2B sandbox execution failed: {str(e)}"
        return Command(
            update={
                "code_execution_result": {"error": str(e)},
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


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
        code_gen_llm = init_chat_model("openai:gpt-5")

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
- For date truncation, use dt.floor(): df['date_column'].dt.floor('D') for daily, 'M' for monthly
- Use dt.strftime() for date formatting: df['date_column'].dt.strftime('%Y-%m-%d')
- Handle timezone-aware dates using dt.tz_localize() or dt.tz_convert()
- For date arithmetic, use pd.Timedelta: df['date_column'] + pd.Timedelta(days=1)

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


class WorkerNodes:
    """Collection of worker nodes for widget processing using create_react_agent."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize worker nodes with LLM."""
        self.llm_model = llm_model

        # Create extended state schema that includes required AgentState fields
        # Define reducers for fields that might be updated concurrently
        def take_last(old, new):
            """Reducer that takes the last (most recent) value"""
            return new if new is not None else old

        def merge_lists(old, new):
            """Reducer that extends lists with new items"""
            if old is None:
                old = []
            if new is None:
                return old
            if isinstance(new, list):
                return old + new
            return old + [new]

        class ExtendedWidgetState(TypedDict):
            # Required AgentState fields for create_react_agent
            messages: Annotated[Sequence[BaseMessage], add_messages]
            remaining_steps: int

            # Custom WidgetAgentState fields (with reducers for concurrent updates)
            task_id: str
            task_status: Annotated[Optional[str], take_last]
            task_instructions: str
            user_prompt: str
            operation: Literal["CREATE", "UPDATE", "DELETE"]
            widget_type: Literal["line", "bar", "pie", "area", "radial", "kpi", "table"]
            widget_id: str
            file_ids: List[str]
            file_sample_data: Annotated[Optional[List[FileSampleData]], take_last]
            file_schemas: Annotated[Optional[List[FileSchema]], take_last]
            title: str
            description: str
            data: Annotated[Optional[Dict[str, Any]], take_last]
            widget_metadata: Annotated[Optional[Dict[str, Any]], take_last]
            data_validated: Annotated[Optional[bool], take_last]
            raw_file_data: Annotated[Optional[Dict[str, Any]], take_last]
            generated_code: Annotated[Optional[str], take_last]
            code_execution_result: Annotated[Optional[Any], take_last]
            error_messages: Annotated[List[str], merge_lists]
            iteration_count: Annotated[Optional[int], take_last]
            current_step: Annotated[Optional[str], take_last]
            widget_supervisor_reasoning: Annotated[Optional[str], take_last]
            created_at: datetime
            updated_at: Annotated[Optional[datetime], take_last]

        # Create the main data processing agent with extended state schema
        self.data_agent = create_react_agent(
            model=llm_model,
            tools=[fetch_data_tool, generate_python_code_tool, e2b_sandbox_tool],
            state_schema=ExtendedWidgetState,
            prompt="""You are a data processing agent for widget creation. Your job is to:
            
1. First, use fetch_data_tool (no parameters needed - extracts file_ids from state)
2. Then, use generate_python_code_tool (no parameters needed - extracts requirements from state) 
3. Finally, use e2b_sandbox_tool (no parameters needed - executes generated code from state)

Always follow this sequence: fetch data → generate code → execute code.
Make sure the final result matches the required format for the widget type.
            """,
        )

        # Create code generation sub-agent
        self.code_generator_agent = create_react_agent(
            model=llm_model,
            tools=[generate_python_code_tool],
            prompt="""You are a specialized Python code generator for data visualization widgets. 
            Generate clean, efficient Python code that transforms raw data into the exact format required for each widget type.
            Focus on data manipulation using pandas and numpy when needed.
            Always set your final result in a variable called 'result'.
            """,
        )

    def data_node(self, state: WidgetAgentState) -> Command:
        """Unified data processing node using create_react_agent with proper state handling."""
        try:
            # Create initial message for the agent
            initial_message = f"""Process data for {state.widget_type} widget:
            
User request: {state.user_prompt}
Task instructions: {state.task_instructions}
Operation: {state.operation}
File IDs: {state.file_ids}
            
Please:
1. Use fetch_data_tool (no parameters needed - extracts file_ids from state)
2. Use generate_python_code_tool (no parameters needed - extracts requirements from state)
3. Use e2b_sandbox_tool (no parameters needed - executes generated code from state)
            """

            # Convert WidgetAgentState to ExtendedWidgetState format
            agent_input = {
                "messages": [{"role": "user", "content": initial_message}],
                "remaining_steps": 10,  # Default step limit
                "task_id": state.task_id,
                "task_status": state.task_status,
                "task_instructions": state.task_instructions,
                "user_prompt": state.user_prompt,
                "operation": state.operation,
                "widget_type": state.widget_type,
                "widget_id": state.widget_id,
                "file_ids": state.file_ids,
                "file_sample_data": state.file_sample_data or [],
                "file_schemas": state.file_schemas or [],
                "title": state.title,
                "description": state.description,
                "data": state.data,
                "widget_metadata": state.widget_metadata,
                "data_validated": state.data_validated,
                "raw_file_data": state.raw_file_data,
                "generated_code": state.generated_code,
                "code_execution_result": state.code_execution_result,
                "error_messages": state.error_messages or [],
                "iteration_count": state.iteration_count,
                "current_step": state.current_step,
                "widget_supervisor_reasoning": state.widget_supervisor_reasoning,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
            }

            # Invoke the create_react_agent - it will handle tool calling with state injection
            agent_result = self.data_agent.invoke(agent_input)

            # The agent result contains the updated state from Command objects
            # Extract what we need for the parent graph
            update_dict = {
                "task_status": "in_progress",
                "updated_at": datetime.now(),
                "iteration_count": state.iteration_count + 1,
            }

            # Extract updated fields from agent result
            # Only extract fields that were actually updated by the tools
            if (
                "raw_file_data" in agent_result
                and agent_result["raw_file_data"] is not None
            ):
                update_dict["raw_file_data"] = agent_result["raw_file_data"]
            if (
                "file_schemas" in agent_result
                and agent_result["file_schemas"] is not None
            ):
                update_dict["file_schemas"] = agent_result["file_schemas"]
            if (
                "file_sample_data" in agent_result
                and agent_result["file_sample_data"] is not None
            ):
                update_dict["file_sample_data"] = agent_result["file_sample_data"]
            if (
                "generated_code" in agent_result
                and agent_result["generated_code"] is not None
            ):
                update_dict["generated_code"] = agent_result["generated_code"]
            if (
                "code_execution_result" in agent_result
                and agent_result["code_execution_result"] is not None
            ):
                update_dict["code_execution_result"] = agent_result[
                    "code_execution_result"
                ]
            if (
                "error_messages" in agent_result
                and agent_result["error_messages"] is not None
            ):
                # For error_messages, we want to merge with existing ones
                existing_errors = state.error_messages or []
                new_errors = agent_result["error_messages"]
                if isinstance(new_errors, list):
                    all_errors = existing_errors + new_errors
                else:
                    all_errors = existing_errors + [new_errors]
                update_dict["error_messages"] = list(
                    set(all_errors)
                )  # Remove duplicates

                # Check if there's an error in the execution result
                if (
                    isinstance(agent_result.get("code_execution_result"), dict)
                    and "error" in agent_result["code_execution_result"]
                ):
                    error_msg = f"Code execution returned error: {agent_result['code_execution_result']['error']}"
                    if error_msg not in update_dict["error_messages"]:
                        update_dict["error_messages"].append(error_msg)

            return Command(goto="widget_supervisor", update=update_dict)

        except Exception as e:
            return Command(
                goto="widget_supervisor",
                update={
                    "error_messages": state.error_messages
                    + [f"Data node error: {str(e)}"],
                    "updated_at": datetime.now(),
                    "task_status": "failed",  # Only update task_status in error case
                },
            )

    def validate_data_node(self, state: WidgetAgentState) -> Command:
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
                "openai:gpt-4o-mini"
            ).with_structured_output(DataValidationResult)

            validation_prompt = f"""
You are a data validation expert. Analyze the generated data against the user requirements and task description.

TASK DESCRIPTION: {state.task_instructions}
USER PROMPT: {state.user_prompt}
WIDGET TYPE: {state.widget_type}
OPERATION: {state.operation}

GENERATED DATA SAMPLE (first 10 items):
{json.dumps(sample_data, indent=2, default=str)}

DATA SCHEMA ANALYSIS:
{json.dumps(data_schema, indent=2)}

VALIDATION CRITERIA:
1. Does the data structure match the widget type requirements?
   - Line/Area charts: dict with 'x' and 'y' arrays
   - Bar charts: dict with 'categories' and 'values' arrays  
   - Pie charts: array of objects with 'label' and 'value' properties
   - Tables: array of objects (records)
   - KPI: dict with 'value' property

2. Does the data content align with the user's request?
3. Are the data types appropriate?
4. Is the data complete and not empty?
5. Does the data make sense for the specified operation?

Provide:
- is_valid: Boolean indicating if data meets all requirements
- confidence_level: 0-100 confidence score
- explanation: Detailed explanation of your assessment
- missing_requirements: List specific missing elements if validation fails
- data_quality_issues: List any data quality problems identified

If confidence is below 80, be very specific about what's wrong and what needs to be fixed.
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
                # Mark task as completed if validation passes
                return Command(
                    goto="widget_supervisor",
                    update={
                        "data_validated": True,
                        "data": state.code_execution_result,
                        "task_status": "completed",
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

    def _get_data_sample(self, data):
        """Extract first 10 rows/items from generated data for validation."""
        if isinstance(data, list):
            return data[:10]
        elif isinstance(data, dict):
            if "x" in data and "y" in data:
                # Chart data - sample both x and y
                return {
                    "x": data["x"][:10] if isinstance(data["x"], list) else data["x"],
                    "y": data["y"][:10] if isinstance(data["y"], list) else data["y"],
                }
            elif "categories" in data and "values" in data:
                # Bar chart data
                return {
                    "categories": data["categories"][:10]
                    if isinstance(data["categories"], list)
                    else data["categories"],
                    "values": data["values"][:10]
                    if isinstance(data["values"], list)
                    else data["values"],
                }
            else:
                # Generic dict - return first 10 key-value pairs
                items = list(data.items())[:10]
                return dict(items)
        else:
            return data  # Return as-is for other types

    def _analyze_data_structure(self, data):
        """Analyze the structure and types of generated data."""
        if isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_type": type(data[0]).__name__ if data else "unknown",
                "sample_keys": list(data[0].keys())
                if data and isinstance(data[0], dict)
                else None,
            }
        elif isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "key_types": {k: type(v).__name__ for k, v in data.items()},
                "array_lengths": {
                    k: len(v) if isinstance(v, list) else None for k, v in data.items()
                },
            }
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100],  # First 100 chars
            }


# Create lazy singleton instance
_worker_nodes_instance = None


def get_worker_nodes():
    """Get or create worker nodes instance."""
    global _worker_nodes_instance
    if _worker_nodes_instance is None:
        _worker_nodes_instance = WorkerNodes()
    return _worker_nodes_instance


# Export individual node functions for graph usage with lazy initialization
def data_node(state: WidgetAgentState) -> Command:
    """Lazy wrapper for data_node."""
    return get_worker_nodes().data_node(state)


def validate_data_node(state: WidgetAgentState) -> Command:
    """Lazy wrapper for validate_data_node."""
    return get_worker_nodes().validate_data_node(state)


# Export tools
__all__ = [
    "fetch_data_tool",
    "e2b_sandbox_tool",
    "generate_python_code_tool",
    "get_chart_config_schema_string",
    "data_node",
    "validate_data_node",
    "get_worker_nodes",
]
