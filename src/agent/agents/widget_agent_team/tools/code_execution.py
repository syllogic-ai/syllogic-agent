"""Code execution tool using E2B sandbox for widget processing."""

import json
import os
import sys
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.models import ChartConfigSchema, WidgetAgentState

from actions.utils import analyze_schema_validation_error, import_config


def create_e2b_sandbox():
    """Create a new E2B Sandbox instance with robust import handling."""
    try:
        # Always use the config module to get the API key properly
        config_module = import_config()
        return config_module.create_e2b_sandbox()
    except ImportError as e:
        # Fallback: try direct import with environment variable
        try:
            import os
            from e2b_code_interpreter import Sandbox
            
            # Get API key from environment
            api_key = os.getenv("E2B_SANDBOX_API_KEY") or os.getenv("E2B_API_KEY")
            if not api_key:
                raise ValueError("E2B_SANDBOX_API_KEY environment variable is required")
            
            return Sandbox(api_key=api_key)
        except Exception as fallback_error:
            raise ImportError(f"Could not create E2B Sandbox. Config import error: {e}. Fallback error: {fallback_error}")


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

                    # Validate the result against ChartConfigSchema
                    try:
                        validated_result = ChartConfigSchema(**parsed_result)
                        # If validation succeeds, use the validated result
                        validated_dict = validated_result.model_dump()

                        success_msg = f"Code executed successfully and passed schema validation. Result: {json.dumps(validated_dict, indent=2)[:500]}{'...' if len(str(validated_dict)) > 500 else ''}"
                        return Command(
                            update={
                                "code_execution_result": validated_dict,
                                "messages": [
                                    ToolMessage(
                                        content=success_msg, tool_call_id=tool_call_id
                                    )
                                ],
                            }
                        )
                    except Exception as validation_error:
                        # Schema validation failed - create detailed error message
                        schema_error_details = analyze_schema_validation_error(
                            parsed_result, validation_error
                        )

                        error_msg = f"Code execution succeeded but result does not match ChartConfigSchema. {schema_error_details}"
                        return Command(
                            update={
                                "code_execution_result": {
                                    "error": "Schema validation failed",
                                    "validation_details": schema_error_details,
                                    "raw_result": parsed_result,
                                },
                                "error_messages": state.error_messages + [error_msg],
                                "messages": [
                                    ToolMessage(
                                        content=error_msg, tool_call_id=tool_call_id
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