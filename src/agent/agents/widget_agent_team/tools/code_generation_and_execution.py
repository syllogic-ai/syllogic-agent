"""Merged Python code generation and execution tool using fully non-blocking concurrent E2B sandbox creation."""

import json
import os
import sys
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.models import ChartConfigSchema, WidgetAgentState
from config import get_langfuse_callback_handler, LANGFUSE_AVAILABLE

from actions.utils import get_chart_config_schema_string, analyze_schema_validation_error
from actions.e2b_sandbox import create_e2b_sandbox, execute_code_in_sandbox, kill_sandbox

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
def generate_and_execute_python_code_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Generates and executes Python code for processing widget data using fully non-blocking concurrent E2B sandbox creation.
    
    This tool performs the complete workflow with fully non-blocking concurrent execution:
    1. Starts E2B sandbox creation and Python code generation concurrently using asyncio.gather()
    2. Uses asyncio.to_thread() for ALL potentially blocking operations
    3. Executes the generated code in the ready E2B sandbox
    4. Validates the results against ChartConfigSchema
    5. Kills the sandbox upon completion
    
    Args:
        state: Current widget agent state containing data and requirements
        tool_call_id: LangGraph tool call identifier
        
    Returns:
        Command: Updated state with execution results or error messages
    """
    
    async def run_fully_async_workflow():
        """Main async workflow that handles all operations."""
        sandbox = None
        generated_code = None
        
        try:
            # Step 1: Prepare concurrent tasks for sandbox creation and code generation
            import logging
            logger = logging.getLogger(__name__)
            
            # Start timing for parallel execution verification
            workflow_start_time = time.time()
            logger.info(f"🚀 STARTING CONCURRENT WORKFLOW at {workflow_start_time:.3f}")
            
            async def create_sandbox_async():
                """Create E2B sandbox in thread pool to avoid blocking."""
                start_time = time.time()
                logger.info(f"🏗️  SANDBOX CREATION STARTED at {start_time:.3f} (offset: {start_time - workflow_start_time:.3f}s)")
                
                result = await asyncio.to_thread(create_e2b_sandbox)
                
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"✅ SANDBOX CREATION COMPLETED at {end_time:.3f} (duration: {duration:.3f}s, offset: {end_time - workflow_start_time:.3f}s)")
                return result
            
            async def generate_code_async():
                """Generate Python code using LLM - all operations in thread pool."""
                start_time = time.time()
                logger.info(f"🧠 CODE GENERATION STARTED at {start_time:.3f} (offset: {start_time - workflow_start_time:.3f}s)")
                
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

                # Get the schema string programmatically - run in thread to avoid blocking
                chart_config_schema = await asyncio.to_thread(get_chart_config_schema_string)
                
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
                
                # Compile the prompt with dynamic variables from Langfuse (REQUIRED) - run in thread to avoid blocking
                code_generation_prompt = await asyncio.to_thread(
                    compile_prompt,
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
                
                # Fetch model configuration from Langfuse (REQUIRED) - run in thread to avoid blocking
                prompt_config = await asyncio.to_thread(
                    get_prompt_config, 
                    "widget_agent_team/data/tools/generate_python_code", 
                    label="latest"
                )
                
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
                
                # Create code generation LLM with Langfuse configuration - run in thread to avoid blocking
                def create_llm():
                    code_gen_llm_params = {
                        "model": model,
                        "temperature": temperature
                    }
                    
                    # Add reasoning_effort if provided (for reasoning models like o1, o3, o4-mini)
                    if reasoning_effort:
                        code_gen_llm_params["reasoning_effort"] = reasoning_effort
                        
                    return ChatOpenAI(**code_gen_llm_params)
                
                code_gen_llm = await asyncio.to_thread(create_llm)

                # Check if there are previous errors to include in the prompt for retry
                error_context = ""
                if state.error_messages:
                    recent_errors = state.error_messages[-2:]  # Include last 2 errors for context
                    error_context = f"""

PREVIOUS ERRORS TO FIX:
{chr(10).join(recent_errors)}

Please fix these issues in your generated code.
"""

                final_prompt = code_generation_prompt_str + error_context
                
                # Create Langfuse callback handler for code generation tracing - run in thread to avoid blocking
                codegen_config = {}
                if LANGFUSE_AVAILABLE:
                    try:
                        langfuse_handler = await asyncio.to_thread(
                            get_langfuse_callback_handler,
                            trace_name="python-code-generation-and-execution",
                            session_id=state.chat_id,
                            user_id=getattr(state, 'user_id', None),
                            tags=["code-generation", "code-execution", "python", "widget", "e2b"],
                            metadata={
                                "dashboard_id": state.dashboard_id,
                                "widget_type": state.widget_type,
                                "operation": state.operation,
                                "tool": "generate_and_execute_python_code_tool"
                            }
                        )
                        if langfuse_handler:
                            codegen_config = {"callbacks": [langfuse_handler]}
                    except Exception as langfuse_error:
                        logger.warning(f"Failed to create Langfuse handler for code generation: {langfuse_error}")
                
                # Invoke LLM with or without tracing (run in thread to avoid blocking)
                def invoke_llm():
                    if codegen_config:
                        return code_gen_llm.invoke(final_prompt, config=codegen_config).content
                    else:
                        return code_gen_llm.invoke(final_prompt).content
                
                generated_code = await asyncio.to_thread(invoke_llm)

                # Clean up the code (remove markdown formatting if present)
                if "```python" in generated_code:
                    generated_code = (
                        generated_code.split("```python")[1].split("```")[0].strip()
                    )
                elif "```" in generated_code:
                    generated_code = generated_code.split("```")[1].split("```")[0].strip()

                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"✅ CODE GENERATION COMPLETED at {end_time:.3f} (duration: {duration:.3f}s, offset: {end_time - workflow_start_time:.3f}s)")
                return generated_code

            # Step 2: Run sandbox creation and code generation concurrently
            concurrent_start_time = time.time()
            logger.info(f"🚀 LAUNCHING CONCURRENT TASKS at {concurrent_start_time:.3f} (offset: {concurrent_start_time - workflow_start_time:.3f}s)")
            logger.info("   → Task 1: E2B Sandbox Creation")
            logger.info("   → Task 2: Python Code Generation")
            logger.info("   → Using asyncio.gather() for true parallel execution")
            
            # Use asyncio.gather to run both tasks concurrently
            concurrent_results = await asyncio.gather(
                create_sandbox_async(),
                generate_code_async(),
                return_exceptions=True
            )
            
            concurrent_end_time = time.time()
            concurrent_duration = concurrent_end_time - concurrent_start_time
            logger.info(f"⏱️  CONCURRENT EXECUTION COMPLETED at {concurrent_end_time:.3f}")
            logger.info(f"   → Total concurrent duration: {concurrent_duration:.3f}s")
            logger.info(f"   → Workflow offset: {concurrent_end_time - workflow_start_time:.3f}s")
            
            # Extract results
            sandbox_result, code_result = concurrent_results
            
            # Check for exceptions
            if isinstance(sandbox_result, Exception):
                error_msg = f"E2B sandbox creation failed: {str(sandbox_result)}"
                return Command(
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                    }
                )
            
            if isinstance(code_result, Exception):
                error_msg = f"Python code generation failed: {str(code_result)}"
                # Kill sandbox if it was created successfully
                if not isinstance(sandbox_result, Exception):
                    cleanup_time = time.time()
                    logger.info(f"🧹 Cleaning up sandbox due to code generation failure at {cleanup_time:.3f}")
                    sandbox_killed = await asyncio.to_thread(kill_sandbox, sandbox_result)
                    logger.info(f"   → Cleanup result: {'✅ Success' if sandbox_killed else '⚠️ Incomplete'}")
                return Command(
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                    }
                )
            
            # Both tasks completed successfully
            sandbox = sandbox_result
            generated_code = code_result
            
            logger.info("🎉 BOTH CONCURRENT TASKS COMPLETED SUCCESSFULLY!")
            logger.info("   → ✅ E2B Sandbox: Created and ready")
            logger.info("   → ✅ Python Code: Generated via LLM")
            logger.info("   → 🚀 True parallel execution achieved with asyncio.gather()")
            concurrent_completion_msg = f"✅ Concurrent execution completed in {concurrent_duration:.3f}s - sandbox ready, code generated"

            # Step 3: Prepare data context for execution
            raw_data = state.raw_file_data or {}

            # Debug: Check if we have raw data
            if not raw_data:
                error_msg = "No raw file data available in state. The fetch_data_tool may not have run successfully."
                # Kill sandbox before returning error
                if sandbox:
                    cleanup_time = time.time()
                    logger.info(f"🧹 Cleaning up sandbox due to missing data at {cleanup_time:.3f}")
                    sandbox_killed = await asyncio.to_thread(kill_sandbox, sandbox)
                    logger.info(f"   → Cleanup result: {'✅ Success' if sandbox_killed else '⚠️ Incomplete'}")
                return Command(
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                    }
                )

            # Prepare the execution context with raw_data and reconstruct DataFrames
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

            # Step 4: Execute context setup in sandbox (run in thread to avoid blocking)
            logger.info(f"📋 STARTING CONTEXT SETUP EXECUTION at {time.time():.3f}")
            logger.info(f"   → Using sandbox instance: {getattr(sandbox, 'id', 'unknown_id')}")
            
            context_start_time = time.time()
            try:
                setup_result = await asyncio.to_thread(execute_code_in_sandbox, sandbox, context_setup, timeout=30.0)
                if not setup_result["success"]:
                    error_msg = f"Context setup failed: {setup_result['error']}"
                    # Kill sandbox before returning error (run in thread to avoid blocking)
                    if sandbox:
                        await asyncio.to_thread(kill_sandbox, sandbox)
                    return Command(
                        update={
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                        }
                    )
                
                context_end_time = time.time()
                context_duration = context_end_time - context_start_time
                logger.info(f"✅ CONTEXT SETUP COMPLETED at {context_end_time:.3f} (duration: {context_duration:.3f}s)")
                if setup_result["logs"]["stdout"]:
                    logger.debug(f"Setup output: {' '.join(setup_result['logs']['stdout'])}")

            except Exception as e:
                context_end_time = time.time()
                context_duration = context_end_time - context_start_time
                logger.error(f"❌ CONTEXT SETUP FAILED at {context_end_time:.3f} (duration: {context_duration:.3f}s)")
                error_msg = f"Context setup execution failed: {str(e)}"
                # Kill sandbox before returning error (run in thread to avoid blocking)
                if sandbox:
                    cleanup_time = time.time()
                    logger.info(f"🧹 Cleaning up sandbox due to context setup error at {cleanup_time:.3f}")
                    sandbox_killed = await asyncio.to_thread(kill_sandbox, sandbox)
                    logger.info(f"   → Cleanup result: {'✅ Success' if sandbox_killed else '⚠️ Incomplete'}")
                return Command(
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                    }
                )

            # Step 5: Execute the main generated code in sandbox (run in thread to avoid blocking)
            logger.info(f"🐍 STARTING MAIN CODE EXECUTION at {time.time():.3f}")
            logger.info(f"   → Using SAME sandbox instance: {getattr(sandbox, 'id', 'unknown_id')}")
            logger.info(f"   → Code length: {len(generated_code)} characters")
            
            main_execution_start_time = time.time()
            try:
                main_result = await asyncio.to_thread(execute_code_in_sandbox, sandbox, generated_code, timeout=60.0)
                if not main_result["success"]:
                    error_msg = f"Python execution error: {main_result['error']}"
                    # Kill sandbox before returning error (run in thread to avoid blocking)
                    if sandbox:
                        await asyncio.to_thread(kill_sandbox, sandbox)
                    return Command(
                        update={
                            "code_execution_result": {"error": main_result['error']},
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                        }
                    )
                
                main_execution_end_time = time.time()
                main_execution_duration = main_execution_end_time - main_execution_start_time
                logger.info(f"✅ MAIN CODE EXECUTION COMPLETED at {main_execution_end_time:.3f} (duration: {main_execution_duration:.3f}s)")
                if main_result["logs"]["stdout"]:
                    logger.debug(f"Main execution output: {' '.join(main_result['logs']['stdout'])}")

            except Exception as e:
                main_execution_end_time = time.time()
                main_execution_duration = main_execution_end_time - main_execution_start_time
                logger.error(f"❌ MAIN CODE EXECUTION FAILED at {main_execution_end_time:.3f} (duration: {main_execution_duration:.3f}s)")
                error_msg = f"Main code execution failed: {str(e)}"
                # Kill sandbox before returning error (run in thread to avoid blocking)
                if sandbox:
                    cleanup_time = time.time()
                    logger.info(f"🧹 Cleaning up sandbox due to main execution error at {cleanup_time:.3f}")
                    sandbox_killed = await asyncio.to_thread(kill_sandbox, sandbox)
                    logger.info(f"   → Cleanup result: {'✅ Success' if sandbox_killed else '⚠️ Incomplete'}")
                return Command(
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                    }
                )

            # Step 6: Extract result variable from sandbox
            logger.info(f"📤 STARTING RESULT EXTRACTION at {time.time():.3f}")
            logger.info(f"   → Using SAME sandbox instance: {getattr(sandbox, 'id', 'unknown_id')}")
            
            extraction_start_time = time.time()
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
                
                extract_result = await asyncio.to_thread(execute_code_in_sandbox, sandbox, extract_result_code, timeout=30.0)
                
                extraction_end_time = time.time()
                extraction_duration = extraction_end_time - extraction_start_time
                logger.info(f"✅ RESULT EXTRACTION COMPLETED at {extraction_end_time:.3f} (duration: {extraction_duration:.3f}s)")
                
                if not extract_result["success"]:
                    error_msg = f"Result extraction failed: {extract_result['error']}"
                    # Kill sandbox before returning error (run in thread to avoid blocking)
                    if sandbox:
                        await asyncio.to_thread(kill_sandbox, sandbox)
                    return Command(
                        update={
                            "code_execution_result": {
                                "error": extract_result['error'],
                                "raw_output": main_result.get("result", "No output captured"),
                            },
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                        }
                    )

                # Get output from extraction
                output_lines = extract_result["logs"]["stdout"]
                output = "\n".join(output_lines) if output_lines else ""

                # Handle case where execution returns no output
                if not output:
                    main_output_lines = main_result["logs"]["stdout"]
                    main_output = "\n".join(main_output_lines) if main_output_lines else "No output captured"
                    
                    error_msg = "E2B sandbox execution returned no output. This usually indicates an API connection issue or the code didn't execute properly."
                    # Kill sandbox before returning error (run in thread to avoid blocking)
                    if sandbox:
                        await asyncio.to_thread(kill_sandbox, sandbox)
                    return Command(
                        update={
                            "code_execution_result": {
                                "error": "No output from sandbox",
                                "raw_output": main_output,
                            },
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                        }
                    )

                # Extract the JSON result from the output
                if "RESULT_START" in output and "RESULT_END" in output:
                    start_idx = output.find("RESULT_START") + len("RESULT_START")
                    end_idx = output.find("RESULT_END")
                    result_json = output[start_idx:end_idx].strip()
                    parsed_result = json.loads(result_json)

                    # Step 7: Validate the result against ChartConfigSchema
                    try:
                        validated_result = ChartConfigSchema(**parsed_result)
                        # If validation succeeds, use the validated result
                        validated_dict = validated_result.model_dump()

                        # Step 8: Kill the sandbox upon successful completion (run in thread to avoid blocking)
                        cleanup_start_time = time.time()
                        logger.info(f"🧹 STARTING SANDBOX CLEANUP at {cleanup_start_time:.3f}")
                        
                        sandbox_killed = await asyncio.to_thread(kill_sandbox, sandbox)
                        
                        cleanup_end_time = time.time()
                        cleanup_duration = cleanup_end_time - cleanup_start_time
                        logger.info(f"🗑️  SANDBOX CLEANUP COMPLETED at {cleanup_end_time:.3f} (duration: {cleanup_duration:.3f}s)")
                        
                        if sandbox_killed:
                            logger.info("   → ✅ Sandbox killed successfully - all resources cleaned up")
                            sandbox_status = "✅ Sandbox killed successfully"
                        else:
                            logger.warning("   → ⚠️  Sandbox cleanup incomplete - may need manual cleanup")
                            sandbox_status = "⚠️ Sandbox cleanup incomplete"

                        # Calculate total workflow time and detailed phase breakdown
                        total_workflow_time = cleanup_end_time - workflow_start_time
                        total_execution_time = context_duration + main_execution_duration + extraction_duration
                        
                        logger.info(f"📊 DETAILED WORKFLOW PERFORMANCE SUMMARY:")
                        logger.info(f"   🚀 PARALLEL PHASE:")
                        logger.info(f"      → Concurrent execution: {concurrent_duration:.3f}s")
                        logger.info(f"      → Sandbox creation: Part of concurrent phase")
                        logger.info(f"      → Code generation: Part of concurrent phase")
                        logger.info(f"   🔄 SEQUENTIAL EXECUTION PHASE:")
                        logger.info(f"      → Context setup: {context_duration:.3f}s")
                        logger.info(f"      → Main code execution: {main_execution_duration:.3f}s")
                        logger.info(f"      → Result extraction: {extraction_duration:.3f}s")
                        logger.info(f"      → Total execution time: {total_execution_time:.3f}s")
                        logger.info(f"   🧹 CLEANUP PHASE:")
                        logger.info(f"      → Sandbox cleanup: {cleanup_duration:.3f}s")
                        logger.info(f"   ⏱️  OVERALL:")
                        logger.info(f"      → Total workflow time: {total_workflow_time:.3f}s")
                        logger.info(f"      → Sandbox reuse: ✅ Same instance used for all executions")

                        success_msg = f"✅ Code generated, executed, and validated successfully!\n\n🚀 PARALLEL EXECUTION VERIFIED:\n{concurrent_completion_msg}\n⚡ E2B sandbox and code generation ran simultaneously\n✅ Code executed in SAME E2B sandbox (reused)\n✅ Results validated against schema\n{sandbox_status}\n\n📊 PERFORMANCE BREAKDOWN:\n• Concurrent phase: {concurrent_duration:.2f}s (sandbox + code gen)\n• Code execution: {main_execution_duration:.2f}s\n• Context setup: {context_duration:.2f}s\n• Result extraction: {extraction_duration:.2f}s\n• Total time: {total_workflow_time:.2f}s\n\nResult: {json.dumps(validated_dict, indent=2)[:500]}{'...' if len(str(validated_dict)) > 500 else ''}"
                        
                        return Command(
                            update={
                                "generated_code": generated_code,
                                "code_execution_result": validated_dict,
                                "messages": [ToolMessage(content=success_msg, tool_call_id=tool_call_id)],
                            }
                        )
                        
                    except Exception as validation_error:
                        # Schema validation failed - create detailed error message
                        schema_error_details = analyze_schema_validation_error(
                            parsed_result, validation_error
                        )

                        error_msg = f"Code execution succeeded but result does not match ChartConfigSchema. {schema_error_details}"
                        # Kill sandbox before returning error (run in thread to avoid blocking)
                        if sandbox:
                            cleanup_time = time.time()
                            logger.info(f"🧹 Cleaning up sandbox due to validation error at {cleanup_time:.3f}")
                            sandbox_killed = await asyncio.to_thread(kill_sandbox, sandbox)
                            logger.info(f"   → Cleanup result: {'✅ Success' if sandbox_killed else '⚠️ Incomplete'}")
                        return Command(
                            update={
                                "generated_code": generated_code,
                                "code_execution_result": {
                                    "error": "Schema validation failed",
                                    "validation_details": schema_error_details,
                                    "raw_result": parsed_result,
                                },
                                "error_messages": state.error_messages + [error_msg],
                                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                            }
                        )
                else:
                    # Could not extract result from code execution
                    main_output_lines = main_result["logs"]["stdout"]
                    main_output = "\n".join(main_output_lines) if main_output_lines else "No output captured"
                    
                    # Return raw output with error context for retry
                    error_msg = f"Could not extract result from code execution. Raw output: {main_output[:500]}{'...' if len(main_output) > 500 else ''}"
                    # Kill sandbox before returning error (run in thread to avoid blocking)
                    if sandbox:
                        await asyncio.to_thread(kill_sandbox, sandbox)
                    return Command(
                        update={
                            "generated_code": generated_code,
                            "code_execution_result": {
                                "error": "No result variable found",
                                "raw_output": main_output,
                            },
                            "error_messages": state.error_messages + [error_msg],
                            "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                        }
                    )

            except Exception as extract_error:
                # If extraction fails, return error with context for potential retry
                main_output_lines = main_result.get("logs", {}).get("stdout", [])
                main_output = "\n".join(main_output_lines) if main_output_lines else "No output captured"

                error_msg = f"Result extraction failed: {str(extract_error)}. This usually means the generated code doesn't end with 'result = final_output'"
                # Kill sandbox before returning error (run in thread to avoid blocking)
                if sandbox:
                    await asyncio.to_thread(kill_sandbox, sandbox)
                return Command(
                    update={
                        "generated_code": generated_code,
                        "code_execution_result": {
                            "error": str(extract_error),
                            "raw_output": main_output,
                        },
                        "error_messages": state.error_messages + [error_msg],
                        "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                    }
                )

        except Exception as e:
            # General exception handler - ensure sandbox is killed
            error_msg = f"Code generation and execution failed: {str(e)}"
            try:
                if 'sandbox' in locals() and sandbox:
                    cleanup_time = time.time()
                    logger.error(f"🧹 Emergency sandbox cleanup due to general exception at {cleanup_time:.3f}")
                    sandbox_killed = await asyncio.to_thread(kill_sandbox, sandbox)
                    logger.info(f"   → Emergency cleanup result: {'✅ Success' if sandbox_killed else '⚠️ Incomplete'}")
            except Exception as cleanup_error:
                logger.error(f"   → Failed to cleanup sandbox: {cleanup_error}")
                pass  # Ensure we always return even if cleanup fails
            return Command(
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                }
            )
    
    # Execute the async workflow in a properly managed event loop
    try:
        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async workflow
        return loop.run_until_complete(run_fully_async_workflow())
        
    except Exception as e:
        # Fallback error handling if even the event loop setup fails
        error_msg = f"Failed to execute async workflow: {str(e)}"
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )