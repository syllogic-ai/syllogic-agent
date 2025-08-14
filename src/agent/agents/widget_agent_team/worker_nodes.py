"""Worker nodes for widget data processing pipeline using create_react_agent."""

import json
from datetime import datetime
from typing import Annotated, Any, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from agent.models import WidgetAgentState

# Initialize Python REPL for code execution
repl = PythonREPL()

# Global variables to store data between agent calls
_fetched_data = None
_generated_code = None
_execution_result = None


@tool
def fetch_data_tool(
    file_ids: Annotated[List[str], "List of file IDs to fetch data from"],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Fetches data from file IDs and returns summary of fetched data for agent context."""
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

        if not file_ids:
            return "Error: No file IDs provided for data fetching"

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

        # Store data in global variable for later access by other tools
        global _fetched_data
        _fetched_data = {
            "raw_file_data": raw_data,
            "file_schemas": schemas_info,
            "file_sample_data": samples_info
        }
        
        return f"Successfully fetched data from {len(file_ids)} files:\n" + "\n".join([
            f"File {info['file_id']}: {info['total_rows']} rows, columns: {', '.join(info['column_names'][:5])}{'...' if len(info['column_names']) > 5 else ''}"
            for info in schemas_info
        ])

    except Exception as e:
        return f"Data fetch error: {str(e)}"


@tool
def python_repl_tool(
    code: Annotated[str, "The Python code to execute"]
) -> str:
    """Executes Python code using Python REPL with fetched data context."""
    try:
        global _fetched_data, _execution_result
        raw_data = _fetched_data.get("raw_file_data", {}) if _fetched_data else {}
        
        # Prepare the execution context with raw_data
        context_setup = f"""
import json
import pandas as pd
import numpy as np
raw_data = {json.dumps(raw_data, default=str)}
"""
        
        # Execute context setup first
        repl.run(context_setup)
        
        # Execute the main code
        result = repl.run(code)
        
        # Try to extract 'result' variable from the REPL session
        try:
            extract_result_code = """
if 'result' in locals():
    print("RESULT_START")
    print(json.dumps(result, default=str))
    print("RESULT_END")
else:
    print("RESULT_START")
    print(json.dumps({"error": "No result variable found"}, default=str))
    print("RESULT_END")
"""
            output = repl.run(extract_result_code)
            
            # Extract the JSON result from the output
            if "RESULT_START" in output and "RESULT_END" in output:
                start_idx = output.find("RESULT_START") + len("RESULT_START")
                end_idx = output.find("RESULT_END")
                result_json = output[start_idx:end_idx].strip()
                parsed_result = json.loads(result_json)
                
                # Store the result globally for the parent graph to access
                _execution_result = parsed_result
                
                return f"Code executed successfully. Result: {json.dumps(parsed_result, indent=2)[:500]}{'...' if len(str(parsed_result)) > 500 else ''}"
            else:
                # Return raw output as result
                result_data = {"output": result}
                _execution_result = result_data
                return f"Code executed with raw output: {str(result)[:500]}{'...' if len(str(result)) > 500 else ''}"
                
        except Exception as extract_error:
            # If extraction fails, return the raw output
            result_data = {"output": result}
            _execution_result = result_data
            return f"Code executed with extraction error: {str(extract_error)}. Raw output: {str(result)[:500]}{'...' if len(str(result)) > 500 else ''}"
            
    except Exception as e:
        error_msg = f"Code execution failed: {str(e)}"
        _execution_result = {"error": str(e)}
        return error_msg


@tool
def generate_python_code_tool(
    widget_type: Annotated[str, "The type of widget (line, bar, pie, etc.)"],
    user_request: Annotated[str, "User's request for what they want to accomplish"],
    operation: Annotated[str, "The operation to perform (CREATE, UPDATE, DELETE)"] = "CREATE"
) -> str:
    """Generates Python code for processing widget data based on user requirements and file schemas."""
    try:
        # Get schema information from global variable if available
        global _fetched_data
        schemas_info = _fetched_data.get("file_schemas", []) if _fetched_data else []
        samples_info = _fetched_data.get("file_sample_data", []) if _fetched_data else []
        
        # Create a code generation agent
        code_gen_llm = init_chat_model("openai:gpt-4o-mini")
        
        code_generation_prompt = f"""Generate Python code to process data for a {widget_type} widget.

USER REQUEST: {user_request}
WIDGET TYPE: {widget_type}
OPERATION: {operation}

FILE SCHEMAS:
{json.dumps(schemas_info, indent=2) if schemas_info else "No schemas available"}

SAMPLE DATA INFO:
{json.dumps(samples_info, indent=2) if samples_info else "No sample data available"}

WIDGET REQUIREMENTS:
- line: Needs 'x' and 'y' arrays or objects with x/y values
- bar: Needs 'categories' (labels) and 'values' (data) arrays
- pie: Needs array of objects with 'label' and 'value' properties
- table: Needs array of objects (rows)
- kpi: Needs object with 'value' and optional 'label', 'change', 'trend'
- area: Similar to line chart with x and y data
- radial: Needs 'categories' and 'values' like bar chart

The code should:
1. Process the raw_data dictionary (keys are file_ids, values are file contents)
2. Manipulate/transform the data according to the user request
3. Return data in the correct format for a {widget_type} widget
4. Set the result in a variable called 'result'
5. Handle errors gracefully with try/except

Generate ONLY the Python code, no explanations or markdown formatting.
Use standard libraries like pandas, numpy if needed (assume they're available).

Example for bar chart:
```python
try:
    # Process the data
    data = raw_data[list(raw_data.keys())[0]]  # Get first file's data
    
    # Transform to widget format
    result = {{
        "categories": ["A", "B", "C"],
        "values": [10, 20, 30]
    }}
except Exception as e:
    result = {{"error": str(e)}}
```
"""
        
        generated_code = code_gen_llm.invoke(code_generation_prompt).content
        
        # Clean up the code (remove markdown formatting if present)
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()
        
        # Store generated code globally for access by other tools
        global _generated_code
        _generated_code = generated_code
        
        return f"Generated Python code for {widget_type} widget:\n```python\n{generated_code}\n```"
        
    except Exception as e:
        error_msg = f"Code generation failed: {str(e)}"
        return error_msg


class WorkerNodes:
    """Collection of worker nodes for widget processing using create_react_agent."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize worker nodes with LLM."""
        self.llm_model = llm_model
        
        # Create the main data processing agent
        self.data_agent = create_react_agent(
            model=llm_model,
            tools=[fetch_data_tool, generate_python_code_tool, python_repl_tool],
            prompt="""You are a data processing agent for widget creation. Your job is to:
            
1. First, fetch data from the provided file IDs using the fetch_data_tool
2. Then, generate appropriate Python code using generate_python_code_tool based on the widget requirements
3. Finally, execute the generated code using python_repl_tool to produce the final widget data

You have access to the current state which contains:
- user_prompt: What the user wants to accomplish
- widget_type: The type of widget (line, bar, pie, table, kpi, area, radial)
- file_ids: List of file IDs to process
- task_instructions: Specific instructions for the task
- operation: The operation to perform

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
        """Unified data processing node using create_react_agent."""
        try:
            # Reset global variables
            global _fetched_data, _generated_code, _execution_result
            _fetched_data = None
            _generated_code = None
            _execution_result = None
            
            # Convert state to messages format expected by the agent
            initial_message = f"""Process data for {state.widget_type} widget:
            
User request: {state.user_prompt}
Task instructions: {state.task_instructions}
Operation: {state.operation}
File IDs: {state.file_ids}
            
Please:
1. Use fetch_data_tool to fetch data from the file IDs: {state.file_ids}
2. Use generate_python_code_tool to generate appropriate Python code for widget type '{state.widget_type}' with user request '{state.user_prompt}'
3. Use python_repl_tool to execute the generated code and produce the final result
            """
            
            # Invoke the data agent
            agent_result = self.data_agent.invoke({
                "messages": [{"role": "user", "content": initial_message}]
            })
            
            # Extract results from global variables set by tools
            update_dict = {
                "task_status": "in_progress",
                "updated_at": datetime.now()
            }
            
            # Update state with data from tools if available
            if _fetched_data:
                update_dict["raw_file_data"] = _fetched_data.get("raw_file_data", {})
                # Convert schemas info back to expected format if needed
                schemas_info = _fetched_data.get("file_schemas", [])
                update_dict["file_schemas"] = schemas_info  # Keep as simplified format for now
                
                samples_info = _fetched_data.get("file_sample_data", [])
                update_dict["file_sample_data"] = samples_info  # Keep as simplified format for now
                
            if _generated_code:
                update_dict["generated_code"] = _generated_code
                
            if _execution_result:
                update_dict["code_execution_result"] = _execution_result
                update_dict["iteration_count"] = state.iteration_count + 1
                
                # Check if there's an error in the result
                if isinstance(_execution_result, dict) and "error" in _execution_result:
                    update_dict["error_messages"] = state.error_messages + [f"Code execution returned error: {_execution_result['error']}"]
            
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