"""Worker nodes for widget data processing pipeline using create_react_agent."""

import json
from datetime import datetime
from typing import Annotated, Any, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from agent.models import WidgetAgentState, FileSchema, ColumnInfo, FileSampleData
from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage

# Initialize Python REPL for code execution
repl = PythonREPL()

def get_fresh_repl():
    """Get a fresh REPL instance to avoid state contamination."""
    return PythonREPL()

# No global variables - use Command objects for immediate state updates


@tool
def fetch_data_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
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
                import sys
                import os
                
                # Add src to path if not already there
                current_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.join(current_dir, '..', '..', '..')  # Go up to src/
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
                    "error_messages": state.error_messages + ["No file IDs provided for data fetching"],
                    "messages": [
                        ToolMessage(
                            content="Error: No file IDs provided for data fetching",
                            tool_call_id=tool_call_id
                        )
                    ]
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
                schemas_info.append({
                    "file_id": file_id,
                    "columns": len(schema_data.get("columns", [])),
                    "total_rows": schema_data.get("total_rows", 0),
                    "column_names": [col["name"] for col in schema_data.get("columns", [])]
                })
                
                sample_data = get_sample_from_file(file_id)
                samples_info.append({
                    "file_id": file_id,
                    "headers": sample_data.get("headers", []),
                    "sample_rows": sample_data.get("sample_rows_returned", 0)
                })

            except Exception as file_error:
                return f"Failed to fetch data for file {file_id}: {str(file_error)}"

        # Convert raw pandas DataFrames to JSON for state storage
        raw_data_json = {}
        for file_id, df in raw_data.items():
            # Convert DataFrame to dict with records orientation
            raw_data_json[file_id] = df.to_dict('records')
        
        # Create FileSchema objects
        file_schemas_objects = [FileSchema(
            file_id=info["file_id"],
            columns=[ColumnInfo(name=col, type="unknown", null_count=0, unique_count=0) 
                    for col in info["column_names"]],
            total_rows=info["total_rows"],
            total_columns=len(info["column_names"])
        ) for info in schemas_info]
        
        # Create FileSampleData objects
        file_sample_objects = [FileSampleData(
            file_id=info["file_id"],
            headers=info["headers"],
            rows=[],  # Could populate from sample data if needed
            total_rows_in_file=schemas_info[i]["total_rows"] if i < len(schemas_info) else 0,
            sample_rows_returned=info["sample_rows"]
        ) for i, info in enumerate(samples_info)]
        
        success_message = f"Successfully fetched data from {len(file_ids)} files:\n" + "\n".join([
            f"File {info['file_id']}: {info['total_rows']} rows, columns: {', '.join(info['column_names'][:5])}{'...' if len(info['column_names']) > 5 else ''}"
            for info in schemas_info
        ])
        
        # Return Command to immediately update state
        return Command(
            update={
                "raw_file_data": raw_data_json,
                "file_schemas": file_schemas_objects,
                "file_sample_data": file_sample_objects,
                "messages": [
                    ToolMessage(
                        content=success_message,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    except Exception as e:
        error_msg = f"Data fetch error: {str(e)}"
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )


@tool
def python_repl_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Executes Python code using Python REPL with state data context."""
    try:
        # Use a fresh REPL instance to avoid contamination from previous executions
        fresh_repl = get_fresh_repl()
        
        # Get the generated code from state
        code = state.generated_code
        if not code:
            return Command(
                update={
                    "error_messages": state.error_messages + ["No generated code to execute"],
                    "messages": [
                        ToolMessage(
                            content="Error: No generated code to execute",
                            tool_call_id=tool_call_id
                        )
                    ]
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
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id
                        )
                    ]
                }
            )
        
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
        setup_result = fresh_repl.run(context_setup)
        print(f"Context setup output: {setup_result}")
        
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
        verification_result = fresh_repl.run(verification_code)
        print(f"Verification output: {verification_result}")
        
        # Execute the main code and capture any execution errors
        try:
            exec_result = fresh_repl.run(code)
            # Check if there were any errors during execution
            if "Error" in exec_result or "Exception" in exec_result or "Traceback" in exec_result:
                error_msg = f"Python execution error: {exec_result}"
                return Command(
                    update={
                        "code_execution_result": {"error": exec_result},
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id
                            )
                        ]
                    }
                )
        except Exception as exec_error:
            error_msg = f"Code execution failed with exception: {str(exec_error)}"
            return Command(
                update={
                    "code_execution_result": {"error": str(exec_error)},
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id
                        )
                    ]
                }
            )
        
        # Try to extract 'result' variable from the REPL session
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
            output = fresh_repl.run(extract_result_code)
            
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
                                content=success_msg,
                                tool_call_id=tool_call_id
                            )
                        ]
                    }
                )
            else:
                # Return raw output with error context for retry
                error_msg = f"Could not extract result from code execution. Raw output: {str(exec_result)[:500]}{'...' if len(str(exec_result)) > 500 else ''}"
                return Command(
                    update={
                        "code_execution_result": {"error": "No result variable found", "raw_output": exec_result},
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_call_id
                            )
                        ]
                    }
                )
                
        except Exception as extract_error:
            # If extraction fails, return error with context for potential retry
            error_msg = f"Result extraction failed: {str(extract_error)}. This usually means the generated code doesn't end with 'result = final_output'"
            return Command(
                update={
                    "code_execution_result": {"error": str(extract_error), "raw_output": exec_result if 'exec_result' in locals() else "No output captured"},
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id
                        )
                    ]
                }
            )
            
    except Exception as e:
        error_msg = f"Code execution failed: {str(e)}"
        return Command(
            update={
                "code_execution_result": {"error": str(e)},
                "error_messages": state.error_messages + [error_msg],
                "messages": [
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )


@tool
def generate_python_code_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
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
        schemas_info = [{
            "file_id": schema.file_id,
            "columns": [col.name for col in schema.columns],
            "total_rows": schema.total_rows
        } for schema in file_schemas] if isinstance(file_schemas, list) else []
        
        samples_info = [{
            "file_id": sample.file_id,
            "headers": sample.headers,
            "sample_rows": sample.sample_rows_returned
        } for sample in file_sample_data] if isinstance(file_sample_data, list) else []
        
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

EXPECTED OUTPUT FORMAT:
- Return executable Python code that processes the DataFrame 'df'
- Code should be optimized for the specified WIDGET TYPE ({widget_type}) and OPERATION ({operation})
- CRITICAL: The final line MUST assign the result to a variable named 'result'
  * ALWAYS end with: result = your_final_data
  * Examples: result = df.groupby('column').sum(), result = monthly_trends, result = final_output
  * DO NOT end with print() statements - use result = instead
- The result should be in the format suitable for the specified widget type:
  * For line/bar charts: dict with 'x' and 'y' arrays or 'categories' and 'values' arrays
  * For pie charts: array of objects with 'label' and 'value' properties
  * For tables: array of objects (records format)
  * For KPI: dict with 'value' property and optional 'change', 'trend' properties
- Ensure the code is ready to execute without additional setup
- OPTIONAL: Start with basic data inspection if needed: print(df.shape), print(df.columns)
- If the DataFrame appears empty or missing expected columns, include error handling

USER REQUEST: {user_request}

AVAILABLE DATA SUMMARY:
- File count: {len(schemas_info)}
- Total rows available: {sum(schema.get('total_rows', 0) for schema in schemas_info)}
- Column information: {', '.join([f"{schema.get('file_id', 'unknown')}: {len(schema.get('columns', []))} columns" for schema in schemas_info])}

REMEMBER: The DataFrame 'df' is already loaded with your data. You can immediately start working with it.

"""
        
        # Check if there are previous errors to include in the prompt for retry
        error_context = ""
        if state.error_messages:
            recent_errors = state.error_messages[-2:]  # Include last 2 errors for context
            error_context = f"""

PREVIOUS ERRORS TO FIX:
{chr(10).join(recent_errors)}

Please fix these issues in your generated code.
"""
        
        final_prompt = code_generation_prompt + error_context
        generated_code = code_gen_llm.invoke(final_prompt).content
        
        # Clean up the code (remove markdown formatting if present)
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        # Return Command with generated code
        success_msg = f"Generated Python code for {widget_type} widget:\n```python\n{generated_code}\n```"
        return Command(
            update={
                "generated_code": generated_code,
                "messages": [
                    ToolMessage(
                        content=success_msg,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
        
    except Exception as e:
        error_msg = f"Code generation failed: {str(e)}"
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id
                    )
                ]
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
            tools=[fetch_data_tool, generate_python_code_tool, python_repl_tool],
            state_schema=ExtendedWidgetState,
            prompt="""You are a data processing agent for widget creation. Your job is to:
            
1. First, use fetch_data_tool (no parameters needed - extracts file_ids from state)
2. Then, use generate_python_code_tool (no parameters needed - extracts requirements from state) 
3. Finally, use python_repl_tool (no parameters needed - executes generated code from state)

Always follow this sequence: fetch data → generate code → execute code.
Make sure the final result matches the required format for the widget type.
            """
        )
        
        # Create code generation sub-agent 
        self.code_generator_agent = create_react_agent(
            model=llm_model,
            tools=[generate_python_code_tool],
            prompt="""You are a specialized Python code generator for data visualization widgets. 
            Generate clean, efficient Python code that transforms raw data into the exact format required for each widget type.
            Focus on data manipulation using pandas and numpy when needed.
            Always set your final result in a variable called 'result'.
            """
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
3. Use python_repl_tool (no parameters needed - executes generated code from state)
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
                "updated_at": state.updated_at
            }
            
            # Invoke the create_react_agent - it will handle tool calling with state injection
            agent_result = self.data_agent.invoke(agent_input)
            
            # The agent result contains the updated state from Command objects
            # Extract what we need for the parent graph
            update_dict = {
                "task_status": "in_progress",
                "updated_at": datetime.now(),
                "iteration_count": state.iteration_count + 1
            }
            
            # Extract updated fields from agent result
            # Only extract fields that were actually updated by the tools
            if "raw_file_data" in agent_result and agent_result["raw_file_data"] is not None:
                update_dict["raw_file_data"] = agent_result["raw_file_data"]
            if "file_schemas" in agent_result and agent_result["file_schemas"] is not None:
                update_dict["file_schemas"] = agent_result["file_schemas"]
            if "file_sample_data" in agent_result and agent_result["file_sample_data"] is not None:
                update_dict["file_sample_data"] = agent_result["file_sample_data"]
            if "generated_code" in agent_result and agent_result["generated_code"] is not None:
                update_dict["generated_code"] = agent_result["generated_code"]
            if "code_execution_result" in agent_result and agent_result["code_execution_result"] is not None:
                update_dict["code_execution_result"] = agent_result["code_execution_result"]
            if "error_messages" in agent_result and agent_result["error_messages"] is not None:
                # For error_messages, we want to merge with existing ones
                existing_errors = state.error_messages or []
                new_errors = agent_result["error_messages"]
                if isinstance(new_errors, list):
                    all_errors = existing_errors + new_errors
                else:
                    all_errors = existing_errors + [new_errors]
                update_dict["error_messages"] = list(set(all_errors))  # Remove duplicates
                
                # Check if there's an error in the execution result
                if (isinstance(agent_result.get("code_execution_result"), dict) and 
                    "error" in agent_result["code_execution_result"]):
                    error_msg = f"Code execution returned error: {agent_result['code_execution_result']['error']}"
                    if error_msg not in update_dict["error_messages"]:
                        update_dict["error_messages"].append(error_msg)
            
            return Command(
                goto="widget_supervisor",
                update=update_dict
            )
            
        except Exception as e:
            return Command(
                goto="widget_supervisor",
                update={
                    "error_messages": state.error_messages + [f"Data node error: {str(e)}"],
                    "updated_at": datetime.now(),
                    "task_status": "failed",  # Only update task_status in error case
                },
            )

    def validate_data_node(self, state: WidgetAgentState) -> Command:
        """Validates the data structure for the specific widget type."""

        def validate_line_chart(data):
            """Validate line chart data structure."""
            if not isinstance(data, dict):
                return False, "Line chart data must be a dictionary"
            if "x" not in data or "y" not in data:
                return False, "Line chart needs 'x' and 'y' properties"
            if not isinstance(data["x"], list) or not isinstance(data["y"], list):
                return False, "Line chart x and y must be arrays"
            if len(data["x"]) != len(data["y"]):
                return False, "Line chart x and y arrays must have same length"
            return True, "Valid line chart data"

        def validate_bar_chart(data):
            """Validate bar chart data structure."""
            if not isinstance(data, dict):
                return False, "Bar chart data must be a dictionary"
            if "categories" not in data or "values" not in data:
                return False, "Bar chart needs 'categories' and 'values' properties"
            if not isinstance(data["categories"], list) or not isinstance(
                data["values"], list
            ):
                return False, "Bar chart categories and values must be arrays"
            if len(data["categories"]) != len(data["values"]):
                return False, "Bar chart categories and values must have same length"
            return True, "Valid bar chart data"

        def validate_pie_chart(data):
            """Validate pie chart data structure."""
            if not isinstance(data, list):
                return False, "Pie chart data must be an array"
            if len(data) == 0:
                return False, "Pie chart data cannot be empty"
            for item in data:
                if not isinstance(item, dict):
                    return False, "Pie chart items must be objects"
                if "label" not in item or "value" not in item:
                    return False, "Pie chart items need 'label' and 'value' properties"
            return True, "Valid pie chart data"

        def validate_table(data):
            """Validate table data structure."""
            if not isinstance(data, list):
                return False, "Table data must be an array"
            if len(data) == 0:
                return False, "Table data cannot be empty"
            return True, "Valid table data"

        def validate_kpi(data):
            """Validate KPI data structure."""
            if not isinstance(data, dict):
                return False, "KPI data must be a dictionary"
            if "value" not in data:
                return False, "KPI needs 'value' property"
            return True, "Valid KPI data"

        # Define validators for each widget type
        widget_validators = {
            "line": validate_line_chart,
            "bar": validate_bar_chart,
            "pie": validate_pie_chart,
            "table": validate_table,
            "kpi": validate_kpi,
            "area": validate_line_chart,  # Same as line chart
            "radial": validate_bar_chart,  # Same as bar chart
        }

        try:
            if state.code_execution_result is None:
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages
                        + ["No execution result to validate"],
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

            validator = widget_validators.get(state.widget_type)
            if not validator:
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages
                        + [
                            f"No validator available for widget type: {state.widget_type}"
                        ],
                        "updated_at": datetime.now(),
                    },
                )

            is_valid, message = validator(state.code_execution_result)

            if is_valid:
                return Command(
                    goto="widget_supervisor",
                    update={
                        "data_validated": True,
                        "data": state.code_execution_result,
                        "updated_at": datetime.now(),
                    },
                )
            else:
                error_msg = (
                    f"Data validation failed for {state.widget_type} widget: {message}"
                )
                return Command(
                    goto="widget_supervisor",
                    update={
                        "data_validated": False,
                        "error_messages": state.error_messages + [error_msg],
                        "updated_at": datetime.now(),
                    },
                )

        except Exception as e:
            return Command(
                goto="widget_supervisor",
                update={
                    "error_messages": state.error_messages
                    + [f"Validation error: {str(e)}"],
                    "updated_at": datetime.now(),
                },
            )

    def update_task_node(self, state: WidgetAgentState) -> Command:
        """Finalizes the task with appropriate status and metadata."""

        # Determine final status
        if state.data_validated and state.data is not None:
            final_status = "completed"
            widget_metadata = {
                "widget_type": state.widget_type,
                "operation": state.operation,
                "data_points": len(state.data) if isinstance(state.data, list) else 1,
                "iterations_needed": state.iteration_count,
                "processing_time": (datetime.now() - state.created_at).total_seconds(),
                "files_processed": len(state.file_ids),
                "success": True,
            }
        else:
            final_status = "failed"
            widget_metadata = {
                "widget_type": state.widget_type,
                "operation": state.operation,
                "failure_reason": state.error_messages[-1]
                if state.error_messages
                else "Unknown error",
                "iterations_attempted": state.iteration_count,
                "files_attempted": len(state.file_ids),
                "success": False,
                "all_errors": state.error_messages,
            }

        return Command(
            goto="widget_supervisor",
            update={
                "task_status": final_status,
                "widget_metadata": widget_metadata,
                "updated_at": datetime.now(),
            },
        )


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

def update_task_node(state: WidgetAgentState) -> Command:
    """Lazy wrapper for update_task_node."""
    return get_worker_nodes().update_task_node(state)

# Export tools
__all__ = [
    "fetch_data_tool", 
    "python_repl_tool",
    "generate_python_code_tool",
    "data_node", 
    "validate_data_node", 
    "update_task_node",
    "get_worker_nodes"
]