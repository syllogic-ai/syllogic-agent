"""Top Level Supervisor Agent using create_react_agent pattern.

This supervisor orchestrates tasks across all specialized agent teams,
analyzes user requests, reads available data, and delegates appropriate tasks.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command, Send
from langgraph.graph import END
from typing_extensions import Annotated

from agent.models import TopLevelSupervisorState, WidgetAgentState, DelegatedTask
from .tools.data_reader import get_available_data
from .tools.task_manager import update_task_status, get_pending_tasks
from .structured_output import SupervisorResponse, TaskCreationPlan, SupervisorDecision, TaskCreationRequest
from actions.prompts import compile_prompt, get_prompt_config

# Handle imports for different execution contexts
try:
    from actions.prompts import retrieve_prompt, get_prompt_config, get_prompt_with_fallback
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.prompts import retrieve_prompt, get_prompt_config, get_prompt_with_fallback

logger = logging.getLogger(__name__)


@tool
def analyze_available_data(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Analyze what data is available for the dashboard. Takes no parameters - uses injected state."""
    try:
        # Check if data analysis has already been completed
        if state.available_data_summary and state.available_files:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content="Data analysis already completed. Available data summary is ready.",
                        tool_call_id=tool_call_id
                    )],
                }
            )
        
        # Get dashboard_id from state
        dashboard_id = state.dashboard_id
        
        data_info = get_available_data(dashboard_id)
        
        success_message = f"""✅ **DATA ANALYSIS COMPLETE**

{data_info['data_summary']}

**📋 NEXT STEP:** Use the plan_widget_tasks tool to intelligently analyze the user request and create the appropriate DelegatedTask objects for the widget_agent_team."""
        
        # Return Command to update state properly
        return Command(
            update={
                "available_files": data_info["available_files"],
                "file_schemas": data_info["file_schemas"],  # Store detailed schema info
                "available_data_summary": data_info["data_summary"],
                "supervisor_status": "delegating",  # Move to next phase
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=success_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error analyzing available data: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )



@tool
def plan_widget_tasks(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """AI-powered tool that analyzes user query and available data to create multiple DelegatedTask objects."""
    try:
        # Check if planning has already been done
        if state.delegated_tasks:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content=f"Task planning already completed. {len(state.delegated_tasks)} task(s) have been planned and are ready for execution.",
                        tool_call_id=tool_call_id
                    )],
                }
            )
        
        # Fetch model configuration and prompt from Langfuse (REQUIRED - no fallbacks)
        try:
            
            # Prepare runtime variables from current state
            prompt_variables = {
                "user_prompt": state.user_prompt,
                "data_summary": state.available_data_summary or "No data summary available",
                "len_available_files": len(state.available_files),
                "available_files": state.available_files,
                "dashboard_id": state.dashboard_id,
                "chat_id": state.chat_id
            }
            
            # Fetch model configuration from Langfuse (REQUIRED)
            logger.info("Fetching model configuration from Langfuse for plan_widget_tasks...")
            prompt_config = get_prompt_config("top_level_supervisor/tools/plan_widget_tasks", label="latest")
            
            # Extract required model and temperature from Langfuse config
            model = prompt_config.get("model")
            temperature = prompt_config.get("temperature")
            
            # Validate required configuration
            if not model:
                raise ValueError("Model configuration is missing or empty in Langfuse prompt config")
            if temperature is None:
                raise ValueError("Temperature configuration is missing in Langfuse prompt config")
            
            logger.info(f"✅ Using Langfuse model config - model: {model}, temperature: {temperature}")
            
            # Compile the prompt with dynamic variables from Langfuse (REQUIRED)
            planning_prompt = compile_prompt(
                "top_level_supervisor/tools/plan_widget_tasks", 
                prompt_variables,
                label="latest"
            )
            
            # Validate compiled prompt (handle different formats)
            if not planning_prompt:
                raise ValueError("Compiled prompt from Langfuse is empty or None")
            
            # Convert to string if needed and validate
            planning_prompt_str = str(planning_prompt)
            if not planning_prompt_str or len(planning_prompt_str.strip()) == 0:
                raise ValueError("Compiled prompt from Langfuse is empty or invalid")
            
            # Use the string version for the LLM
            planning_prompt = planning_prompt_str
            
            logger.info(f"✅ Successfully compiled Langfuse planning prompt with {len(prompt_variables)} variables")
            
        except Exception as e:
            error_msg = f"Error in AI planning - failed to fetch prompt/config from Langfuse: {str(e)}"
            logger.error(error_msg)
            return Command(
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                }
            )

        # Initialize planning LLM with Langfuse configuration
        planning_llm = ChatOpenAI(model=model, temperature=temperature)
        planning_llm_structured = planning_llm.with_structured_output(TaskCreationPlan)

        # Get structured AI planning response
        task_plan = planning_llm_structured.invoke(planning_prompt)
        
        # Create DelegatedTask objects from the AI plan
        created_task_names = []
        updated_tasks = state.delegated_tasks.copy()  # Start with existing tasks
        
        for task_request in task_plan.tasks:
            # Create each task directly using DelegatedTask model
            new_task = DelegatedTask(
                target_agent="widget_agent_team",
                task_instructions=task_request.task_instructions,
                widget_type=task_request.widget_type,
                operation=task_request.operation,
                file_ids=task_request.file_ids,
                title=task_request.title,
                description=task_request.description,
                user_prompt=state.user_prompt,
                dashboard_id=state.dashboard_id,
                chat_id=state.chat_id,
                widget_id=task_request.widget_id,
                task_status="pending"  # Always create as pending
            )
            updated_tasks.append(new_task)
            created_task_names.append(f"{task_request.title} ({task_request.widget_type})")
        
        # Create comprehensive success message
        success_message = f"""✅ **AI PLANNING COMPLETE**

**Strategy:** {task_plan.strategy_summary}

**Tasks Created:** {len(task_plan.tasks)} 
{chr(10).join([f"• {name}" for name in created_task_names])}

**Duplicate Check:** {task_plan.duplicate_check}

**📋 NEXT STEP:** Use execute_widget_tasks to process these AI-planned tasks with the widget_agent_team."""

        return Command(
            update={
                "delegated_tasks": updated_tasks,  # CRITICAL: Include updated tasks in state
                "current_reasoning": f"AI planning completed: {len(task_plan.tasks)} widget tasks created based on intelligent analysis of user request and available data.",
                "supervisor_status": "monitoring",  # Ready to monitor task execution
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=success_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error in AI planning: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


# delegate_widget_task REMOVED - redundant with plan_widget_tasks + execute_widget_tasks workflow


@tool
def check_task_status(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Check the status of all delegated tasks."""
    try:
        if not state.delegated_tasks:
            message = "No tasks have been delegated yet."
            return Command(
                update={
                    "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
                }
            )
        
        status_report = []
        status_report.append(f"Total tasks: {len(state.delegated_tasks)}")
        
        pending_tasks = get_pending_tasks(state)
        in_progress_tasks = [t for t in state.delegated_tasks if t.task_status == "in_progress"]
        completed_tasks = [t for t in state.delegated_tasks if t.task_status == "completed"]
        failed_tasks = [t for t in state.delegated_tasks if t.task_status == "failed"]
        
        status_report.append(f"Pending: {len(pending_tasks)}")
        status_report.append(f"In Progress: {len(in_progress_tasks)}")
        status_report.append(f"Completed: {len(completed_tasks)}")
        status_report.append(f"Failed: {len(failed_tasks)}")
        
        # Check if all tasks are complete
        all_completed = len(completed_tasks) == len(state.delegated_tasks) and len(state.delegated_tasks) > 0
        if all_completed:
            status_report.append("✅ All tasks are completed!")
        
        status_message = "\n".join(status_report)
        
        return Command(
            update={
                "all_tasks_completed": all_completed,
                "messages": [ToolMessage(content=status_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error checking task status: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


@tool
def execute_widget_tasks(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Execute pending widget tasks by directly invoking widget_agent_team subgraph."""
    try:
        # Find pending widget tasks
        pending_tasks = [
            task for task in state.delegated_tasks 
            if task.target_agent == "widget_agent_team" and task.task_status == "pending"
        ]
        
        if not pending_tasks:
            message = "No pending widget tasks to execute"
            return Command(
                update={
                    "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
                }
            )
        
        # Process the first pending task (one at a time for now)
        current_task = pending_tasks[0]
        
        # Import widget graph here to avoid circular imports
        from agent.graph import build_widget_agent_graph
        
        # Create widget agent input from DelegatedTask data
        widget_input = {
            "messages": [],  # Required for LangGraph
            "remaining_steps": 10,  # Required for LangGraph
            "task_id": current_task.task_id,
            "task_instructions": current_task.task_instructions,
            "user_prompt": current_task.user_prompt,
            "operation": current_task.operation,
            "widget_type": current_task.widget_type,
            "dashboard_id": current_task.dashboard_id,
            "chat_id": current_task.chat_id,
            "file_ids": current_task.file_ids,
            "widget_id": current_task.widget_id or str(uuid.uuid4()),  # Generate ID if None
            "title": current_task.title,
            "description": current_task.description,
            "task_status": "in_progress",
            "created_at": current_task.created_at,
            "updated_at": datetime.now(),
            "started_at": datetime.now(),
            "iteration_count": 0,
            "file_schemas": [],
            "file_sample_data": [],
            "error_messages": [],
            "widget_creation_completed": False,
            "widget_update_completed": False,
            "widget_deletion_completed": False,
            "data_validated": False,
        }
        
        logger.info(f"Invoking widget_agent_team for task {current_task.task_id}: {current_task.operation} {current_task.widget_type} widget")
        
        # Build and invoke widget agent subgraph
        widget_graph = build_widget_agent_graph()
        widget_result = widget_graph.invoke(widget_input)
        
        # Update task status based on widget result
        updated_tasks = []
        for task in state.delegated_tasks:
            if task.task_id == current_task.task_id:
                # Update task with result
                task.task_status = widget_result.get("task_status", "completed")
                task.completed_at = datetime.now()
                task.started_at = datetime.now()
                
                if widget_result.get("error_messages"):
                    task.error_message = "; ".join(widget_result["error_messages"])
                    task.task_status = "failed"
                else:
                    task.result = f"Widget {current_task.operation} operation completed successfully"
                    
            updated_tasks.append(task)
        
        execution_message = f"✅ **WIDGET TASK EXECUTED**\n\nCompleted {current_task.operation} {current_task.widget_type} widget task\nStatus: {widget_result.get('task_status', 'completed')}"
        
        logger.info(f"Widget task {current_task.task_id} completed with status: {widget_result.get('task_status')}")
        
        return Command(
            update={
                "delegated_tasks": updated_tasks,
                "supervisor_status": "analyzing",  # Return to analyzing to check for more tasks
                "current_reasoning": f"Widget task completed: {widget_result.get('task_status', 'completed')}",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=execution_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error executing widget tasks: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "supervisor_status": "failed",
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


@tool
def generate_error_response(
    failed_tool: str,
    error_messages: List[str],
    user_prompt: str,
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Generate an AI-powered error response when tools fail repeatedly."""
    try:
        # Create a user-friendly error message
        error_summary = f"I encountered repeated issues while trying to {failed_tool}. "
        
        if failed_tool == "analyze_available_data":
            error_summary += "I was unable to access or analyze the data files needed for your request. "
        elif failed_tool == "plan_widget_tasks":
            error_summary += "I had trouble planning the appropriate widgets for your request. "
        elif failed_tool == "delegate_widget_task":
            error_summary += "I couldn't successfully delegate the widget creation task. "
        else:
            error_summary += "There was a technical issue with the system. "
        
        # Add specific error details if available
        if error_messages:
            recent_errors = error_messages[-3:]  # Show last 3 errors
            error_details = "\\n".join([f"• {error}" for error in recent_errors])
            error_summary += f"\\n\\nTechnical details:\\n{error_details}"
        
        error_summary += "\\n\\nPlease try your request again later, or contact support if the issue persists."
        
        return Command(
            goto=END,
            update={
                "final_response": error_summary,
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=error_summary, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        fallback_message = f"I'm sorry, but I encountered technical difficulties while processing your request: '{user_prompt}'. Please try again later or contact support."
        logger.error(f"Error generating error response: {e}")
        return Command(
            goto=END,
            update={
                "final_response": fallback_message,
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=fallback_message, tool_call_id=tool_call_id)],
            }
        )


@tool
def finalize_response(
    final_message: str,
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Finalize the response when all tasks are completed."""
    try:
        completion_message = f"Response finalized: {final_message}"
        
        logger.info("Top level supervisor completed all tasks")
        
        return Command(
            goto=END,
            update={
                "final_response": final_message,
                "supervisor_status": "completed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=completion_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error finalizing response: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


# Define the tools available to the supervisor
supervisor_tools = [
    analyze_available_data,
    plan_widget_tasks,
    execute_widget_tasks,
    check_task_status,
    generate_error_response,
    finalize_response
]

# Create the supervisor agent using create_react_agent with structured output
def create_top_level_supervisor(model_name: Optional[str] = None):
    """Create the top-level supervisor agent with Langfuse prompt and model configuration.
    
    This function REQUIRES Langfuse to be available and will fail if prompts or model 
    configuration cannot be fetched from Langfuse.
    
    Raises:
        RuntimeError: If Langfuse is not available or prompts cannot be fetched
        ValueError: If fetched configuration is invalid or incomplete
    """
    
    # Check if Langfuse is available
    try:
        from config import LANGFUSE_AVAILABLE
        if not LANGFUSE_AVAILABLE:
            raise RuntimeError(
                "Langfuse is not available. Cannot create top_level_supervisor without Langfuse integration. "
                "Please install langfuse: pip install langfuse"
            )
    except ImportError:
        raise RuntimeError(
            "Cannot import Langfuse configuration. Please ensure langfuse is installed and configured."
        )
    
    try:
        # Fetch prompt from Langfuse (REQUIRED)
        logger.info("Fetching top_level_supervisor prompt from Langfuse...")
        langfuse_prompt = retrieve_prompt("top_level_supervisor/top_level_supervisor", label="latest")
        
        # Handle different prompt formats (string or chat messages)
        # Note: We get the raw prompt template here - variables will be injected at runtime
        if hasattr(langfuse_prompt, 'prompt'):
            prompt_content = langfuse_prompt.prompt
            # If it's a list (chat messages), convert to string
            if isinstance(prompt_content, list):
                system_prompt = "\n".join([msg.get('content', str(msg)) for msg in prompt_content])
            else:
                system_prompt = str(prompt_content)
        else:
            system_prompt = str(langfuse_prompt)
        
        # Store the raw langfuse prompt for runtime compilation with variables
        raw_langfuse_prompt = langfuse_prompt
        
        # Ensure prompt is a string before validation
        if not isinstance(system_prompt, str):
            logger.warning(f"System prompt is not a string, converting: {type(system_prompt)}")
            if isinstance(system_prompt, list):
                system_prompt = "\n".join([str(item) for item in system_prompt])
            else:
                system_prompt = str(system_prompt)
        
        # Validate prompt content
        if not system_prompt or len(system_prompt.strip()) == 0:
            raise ValueError("Retrieved prompt from Langfuse is empty or invalid")
            
    except Exception as e:
        error_msg = f"Failed to fetch prompt 'top_level_supervisor' from Langfuse: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    try:
        # Fetch model configuration from Langfuse (REQUIRED)
        logger.info("Fetching model configuration from Langfuse...")
        prompt_config = get_prompt_config("top_level_supervisor/top_level_supervisor", label="latest")
        
        # Extract required configuration
        model = prompt_config.get("model")
        temperature = prompt_config.get("temperature")
        
        # Validate configuration
        if not model:
            raise ValueError("Model configuration is missing or empty in Langfuse prompt config")
        if temperature is None:
            raise ValueError("Temperature configuration is missing in Langfuse prompt config")
            
        # Override model if explicitly provided
        if model_name:
            logger.info(f"Overriding Langfuse model '{model}' with provided model '{model_name}'")
            model = model_name
            
        logger.info(f"✅ Using Langfuse configuration - model: {model}, temperature: {temperature}")
        
    except Exception as e:
        error_msg = f"Failed to fetch model configuration from Langfuse: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # system_prompt is already validated and converted to string above
    
    try:
        # Initialize the language model with Langfuse configuration
        llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )
        
        logger.info(f"Creating supervisor with prompt length: {len(system_prompt)}")
        
    except Exception as e:
        error_msg = f"Failed to initialize ChatOpenAI with Langfuse configuration (model: {model}, temperature: {temperature}): {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # Create the react agent with Langfuse-managed prompt
    return create_react_agent(
        model=llm,
        tools=supervisor_tools,
        prompt=system_prompt,
        state_schema=TopLevelSupervisorState  # Specify the state schema for InjectedState
        # Temporarily removing response_format to avoid validation errors
        # response_format=SupervisorResponse
    )


def handle_tool_failure(supervisor_state: TopLevelSupervisorState, error_message: str) -> Dict[str, Any]:
    """Handle tool failures with retry logic and error response generation."""
    try:
        # Extract tool name from error message if possible
        failed_tool = "unknown_tool"
        if "analyze_available_data" in error_message:
            failed_tool = "analyze_available_data"
        elif "plan_widget_tasks" in error_message:
            failed_tool = "plan_widget_tasks"
        elif "delegate_widget_task" in error_message:
            failed_tool = "delegate_widget_task"
        
        # Update failure counts
        current_failures = supervisor_state.tool_failure_counts.get(failed_tool, 0) + 1
        supervisor_state.tool_failure_counts[failed_tool] = current_failures
        supervisor_state.last_failed_tool = failed_tool
        
        # Add error to messages
        supervisor_state.error_messages.append(f"Tool {failed_tool} failed (attempt {current_failures}): {error_message}")
        
        # Check if we've exceeded retry limit
        if current_failures > supervisor_state.max_tool_retries:
            # Generate AI-powered error response
            error_summary = f"I encountered repeated issues with {failed_tool} after {current_failures} attempts. "
            
            if failed_tool == "analyze_available_data":
                error_summary += "I was unable to access or analyze the data files needed for your request. This might be due to database connectivity issues or missing data files."
            elif failed_tool == "plan_widget_tasks":
                error_summary += "I had trouble planning the appropriate widgets for your request. This might be due to complex requirements or system limitations."
            elif failed_tool == "delegate_widget_task":
                error_summary += "I couldn't successfully delegate the widget creation task. This might be due to resource constraints or system issues."
            else:
                error_summary += "There was a persistent technical issue with the system."
            
            error_summary += f"\\n\\nYour request: '{supervisor_state.user_prompt}'\\n\\nPlease try again later or contact support if the issue persists."
            
            return {
                "final_response": error_summary,
                "supervisor_status": "failed",
                "error_messages": supervisor_state.error_messages,
                "tool_failure_counts": supervisor_state.tool_failure_counts,
                "updated_at": datetime.now()
            }
        else:
            # Tool can still be retried
            retry_message = f"Tool {failed_tool} failed (attempt {current_failures}/{supervisor_state.max_tool_retries}). Will attempt retry if appropriate."
            
            return {
                "current_reasoning": retry_message,
                "supervisor_status": supervisor_state.supervisor_status,  # Keep current status for retry
                "error_messages": supervisor_state.error_messages,
                "tool_failure_counts": supervisor_state.tool_failure_counts,
                "last_failed_tool": failed_tool,
                "updated_at": datetime.now()
            }
            
    except Exception as e:
        logger.error(f"Error in handle_tool_failure: {e}")
        return {
            "final_response": f"I encountered technical difficulties while processing your request. Please try again later.",
            "supervisor_status": "failed",
            "error_messages": supervisor_state.error_messages + [f"Error handler failed: {str(e)}"],
            "updated_at": datetime.now()
        }


def top_level_supervisor(state) -> Dict[str, Any]:
    """Main entry point for the top level supervisor node.
    
    This function creates and invokes the supervisor agent, handling the
    orchestration of tasks across specialized agent teams.
    """
    supervisor_state = None  # Initialize to avoid UnboundLocalError
    try:
        # Convert dict state to TopLevelSupervisorState for proper tool injection
        if isinstance(state, dict):
            # Convert 'pending' status to 'analyzing' if present
            if state.get('supervisor_status') == 'pending':
                state['supervisor_status'] = 'analyzing'
            supervisor_state = TopLevelSupervisorState(**state)
        else:
            supervisor_state = state
            
        # Create the supervisor agent
        supervisor_agent = create_top_level_supervisor()
        
        # Prepare the message for the agent
        user_message = HumanMessage(
            content=f"""
User Request: {supervisor_state.user_prompt}
Dashboard ID: {supervisor_state.dashboard_id}
Chat ID: {supervisor_state.chat_id}
Request ID: {supervisor_state.request_id}
User ID: {supervisor_state.user_id}

Please analyze this request and coordinate the necessary tasks to fulfill it.
Start by analyzing the available data, then determine what needs to be done.
"""
        )
        
        # For InjectedState to work properly with create_react_agent, 
        # we need to pass the full state as the primary input
        agent_input = {
            **supervisor_state.model_dump(),  # Include all state fields
            "messages": [user_message],  # Add the user message to messages
        }
        
        # Invoke the agent with the complete state and handle potential failures
        try:
            result = supervisor_agent.invoke(agent_input)
        except Exception as tool_error:
            # Handle tool execution errors
            return handle_tool_failure(supervisor_state, str(tool_error))
        
        # Check if any tool in the result requested termination via Command(goto=END)
        # This happens when finalize_response or generate_error_response tools are called
        if result and "messages" in result:
            for message in result["messages"]:
                # Check if this is a ToolMessage from finalize_response or generate_error_response
                if hasattr(message, 'tool_call_id') and hasattr(message, 'content'):
                    if ("Response finalized:" in str(message.content) or 
                        "I encountered repeated issues" in str(message.content) or
                        "I'm sorry, but I encountered technical difficulties" in str(message.content)):
                        # These tools used goto=END, so we should terminate
                        logger.info("Tool requested termination - ending supervisor execution")
                        from langgraph.types import Command
                        return Command(goto=END, update=result)
        
        # Extract structured response
        structured_response = None
        response_content = "Task analysis completed"
        
        if result and "structured_response" in result:
            structured_response = result["structured_response"]
            response_content = structured_response.message
            
            # Update supervisor status based on structured response
            if structured_response.status in ["analyzing", "planning", "delegating", "monitoring", "completed", "failed"]:
                supervisor_state.supervisor_status = structured_response.status
                
        elif result and "messages" in result:
            # Fallback to message content if no structured response
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_content = last_message.content
            else:
                response_content = str(last_message)
        
        # Check if supervisor status indicates completion - if so, terminate
        if supervisor_state.supervisor_status in ["completed", "failed"]:
            logger.info(f"Supervisor status is {supervisor_state.supervisor_status} - terminating execution")
            from langgraph.types import Command
            return Command(
                goto=END,
                update={
                    "current_reasoning": response_content,
                    "supervisor_status": supervisor_state.supervisor_status,
                    "updated_at": datetime.now(),
                    "structured_response": structured_response
                }
            )
        
        # Update state
        supervisor_state.current_reasoning = response_content
        supervisor_state.updated_at = datetime.now()
        
        # Return both regular state updates and structured response for external use
        return {
            "current_reasoning": response_content,
            "supervisor_status": supervisor_state.supervisor_status,
            "updated_at": supervisor_state.updated_at,
            "structured_response": structured_response
        }
        
    except Exception as e:
        logger.error(f"Error in top level supervisor: {e}")
        # Handle case where supervisor_state creation failed
        if supervisor_state is None:
            if isinstance(state, dict):
                error_messages = state.get("error_messages", []) + [f"Supervisor error: {str(e)}"]
            else:
                error_messages = [f"Supervisor error: {str(e)}"]
            
            return {
                "error_messages": error_messages,
                "supervisor_status": "failed",
                "updated_at": datetime.now()
            }
        else:
            supervisor_state.error_messages.append(f"Supervisor error: {str(e)}")
            supervisor_state.supervisor_status = "failed"
            supervisor_state.updated_at = datetime.now()
            
            return {
                "error_messages": supervisor_state.error_messages,
                "supervisor_status": "failed",
                "updated_at": supervisor_state.updated_at
            }