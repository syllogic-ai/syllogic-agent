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

from agent.models import TopLevelSupervisorState, WidgetAgentState, DelegatedTask, TaskDependency
from .tools.data_reader import get_available_data
from .tools.task_manager import update_task_status, get_pending_tasks
from .tools.database_operations import finalize_created_widgets
from .structured_output import SupervisorResponse, TaskCreationPlan, SupervisorDecision, TaskCreationRequest
from actions.prompts import compile_prompt, get_prompt_config
from actions.tasks import create_tasks_from_delegated_tasks, generate_task_group_id, format_task_list_message, update_task_status as update_db_task_status
from actions.messages import create_task_list_message
from agent.models import UpdateTaskInput
from config import get_langfuse_callback_handler, LANGFUSE_AVAILABLE, get_supabase_client

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
            reasoning_effort = prompt_config.get("reasoning_effort")
            
            # Validate required configuration
            if not model:
                raise ValueError("Model configuration is missing or empty in Langfuse prompt config")
            if temperature is None:
                raise ValueError("Temperature configuration is missing in Langfuse prompt config")
            
            logger.info(f"✅ Using Langfuse model config - model: {model}, temperature: {temperature}, reasoning_effort: {reasoning_effort}")
            
            # Compile the prompt with dynamic variables from Langfuse (REQUIRED)
            planning_prompt = compile_prompt(
                "top_level_supervisor/tools/plan_widget_tasks", 
                prompt_variables,
                label="latest"
            )
            
            # Validate compiled prompt (handle different formats)
            if not planning_prompt:
                raise ValueError("Compiled prompt from Langfuse is empty or None")
            
            # Handle chat message format from Langfuse
            if isinstance(planning_prompt, list) and len(planning_prompt) > 0:
                # Extract content from chat message format
                first_message = planning_prompt[0]
                if isinstance(first_message, dict) and 'content' in first_message:
                    planning_prompt_content = first_message['content']
                else:
                    planning_prompt_content = str(first_message)
            else:
                planning_prompt_content = str(planning_prompt)
            
            if not planning_prompt_content or len(planning_prompt_content.strip()) == 0:
                raise ValueError("Compiled prompt from Langfuse is empty or invalid")
            
            # Use the extracted content for the LLM
            planning_prompt = planning_prompt_content
            
            logger.info(f"✅ Extracted planning prompt content: {len(planning_prompt_content)} characters")
            
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
        planning_llm_params = {
            "model": model,
            "temperature": temperature
        }
        
        # Add reasoning_effort if provided (for reasoning models like o1, o3, o4-mini)
        if reasoning_effort:
            planning_llm_params["reasoning_effort"] = reasoning_effort
            
        planning_llm = ChatOpenAI(**planning_llm_params)
        planning_llm_structured = planning_llm.with_structured_output(TaskCreationPlan)

        # Create Langfuse callback handler for planning LLM tracing
        planning_config = {}
        if LANGFUSE_AVAILABLE:
            try:
                langfuse_handler = get_langfuse_callback_handler(
                    trace_name="widget-task-planning",
                    session_id=state.chat_id,
                    user_id=state.user_id,
                    tags=["planning", "widget-tasks", "llm"],
                    metadata={
                        "dashboard_id": state.dashboard_id,
                        "request_id": state.request_id,
                        "tool": "plan_widget_tasks"
                    }
                )
                if langfuse_handler:
                    planning_config = {"callbacks": [langfuse_handler]}
            except Exception as langfuse_error:
                logger.warning(f"Failed to create Langfuse handler for planning LLM: {langfuse_error}")

        # Get structured AI planning response
        if planning_config:
            task_plan = planning_llm_structured.invoke(planning_prompt, config=planning_config)
        else:
            task_plan = planning_llm_structured.invoke(planning_prompt)
        
        # Use request_id directly as task_group_id (no generation needed)
        task_group_id = state.request_id
        
        # Create DelegatedTask objects from the AI plan
        created_task_names = []
        updated_tasks = state.delegated_tasks.copy()  # Start with existing tasks
        updated_task_dependencies = state.task_dependencies.copy()  # Start with existing dependencies
        
        for i, task_request in enumerate(task_plan.tasks):
            # Create each task directly using DelegatedTask model
            new_task = DelegatedTask(
                target_agent="widget_agent_team",
                task_instructions=task_request.task_instructions,
                task_title=task_request.title,  # Set task_title for database mapping
                widget_type=task_request.widget_type,
                operation=task_request.operation,
                file_ids=task_request.file_ids,
                title=task_request.title,
                description=task_request.description,
                user_prompt=state.user_prompt,
                dashboard_id=state.dashboard_id,
                chat_id=state.chat_id,
                widget_id=task_request.widget_id,
                reference_widget_id=[],  # Start empty, will be populated when dependencies resolve
                task_status="pending",  # Always create as pending
                task_group_id=task_group_id  # Set task group ID
            )
            updated_tasks.append(new_task)
            created_task_names.append(f"{task_request.title} ({task_request.widget_type})")
            
            # Create dependency tracking for text blocks that should reference other tasks
            if (task_request.widget_type == "text" and 
                any(keyword in task_request.task_instructions.lower() for keyword in 
                    ["reference", "explain", "analyze", "describe", "describe the chart", "describe the bar chart", "seasonal", "trend"])):
                
                # Find which tasks this text block should depend on (typically chart tasks created before it)
                dependent_task_ids = []
                for j, other_task in enumerate(task_plan.tasks):
                    if (j < i and  # Only tasks created before this text block
                        other_task.widget_type in ["bar", "line", "pie", "area", "radial"]):  # Chart types
                        # Find the corresponding task ID from our created tasks
                        other_delegated_task = updated_tasks[len(state.delegated_tasks) + j]
                        dependent_task_ids.append(other_delegated_task.task_id)
                        logger.info(f"Text block task {new_task.task_id} will depend on {other_task.widget_type} task {other_delegated_task.task_id}")
                
                if dependent_task_ids:
                    # Create dependency tracking entry
                    task_dependency = TaskDependency(
                        task_id=new_task.task_id,
                        dependent_on=dependent_task_ids,
                        reference_widget_ids=[]  # Will be populated when dependent tasks complete
                    )
                    updated_task_dependencies.append(task_dependency)
                    logger.info(f"Created dependency tracking for task {new_task.task_id} dependent on tasks: {dependent_task_ids}")
        
        # Create database tasks and task list message
        try:
            supabase = get_supabase_client()
            
            # Get only the new tasks that were just created
            new_tasks_count = len(updated_tasks) - len(state.delegated_tasks)
            new_tasks = updated_tasks[-new_tasks_count:] if new_tasks_count > 0 else []
            
            logger.info(f"Creating {len(new_tasks)} database tasks (total tasks: {len(updated_tasks)}, previous: {len(state.delegated_tasks)})")
            
            # Create database tasks from delegated tasks
            db_tasks = create_tasks_from_delegated_tasks(
                supabase, 
                new_tasks,  # Only new tasks
                state.chat_id, 
                state.dashboard_id, 
                task_group_id
            )
            
            # Format task list message content
            task_list_content = format_task_list_message(db_tasks, task_group_id)
            
            # Create task list message in chat
            task_message = create_task_list_message(
                supabase,
                state.chat_id,
                task_group_id,
                task_list_content
            )
            
            logger.info(f"Created {len(db_tasks)} database tasks and task list message {task_message.id}")
            
        except Exception as db_error:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Failed to create database tasks/message: {db_error}")
            logger.error(f"Full traceback: {error_details}")
            # Continue with execution even if database operations fail
        
        # Create comprehensive success message
        success_message = f"""✅ **AI PLANNING COMPLETE**

**Strategy:** {task_plan.strategy_summary}

**Tasks Created:** {len(task_plan.tasks)} 
{chr(10).join([f"• {name}" for name in created_task_names])}

**Duplicate Check:** {task_plan.duplicate_check}

**Task Group ID:** {task_group_id}

**📋 NEXT STEP:** Use execute_widget_tasks to process these AI-planned tasks with the widget_agent_team."""

        return Command(
            update={
                "delegated_tasks": updated_tasks,  # CRITICAL: Include updated tasks in state
                "task_dependencies": updated_task_dependencies,  # CRITICAL: Include dependency tracking
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


def _generate_task_status_report(
    state: TopLevelSupervisorState, 
    tool_call_id: str
) -> Command:
    """Helper function to generate task status report (consolidated from check_task_status)."""
    if not state.delegated_tasks:
        message = "No tasks have been delegated yet."
        return Command(
            update={
                "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
            }
        )
    
    status_report = []
    status_report.append(f"📊 **TASK STATUS REPORT**")
    status_report.append(f"Total tasks: {len(state.delegated_tasks)}")
    
    pending_tasks = get_pending_tasks(state)
    in_progress_tasks = [t for t in state.delegated_tasks if t.task_status == "in_progress"]
    completed_tasks = [t for t in state.delegated_tasks if t.task_status == "completed"]
    failed_tasks = [t for t in state.delegated_tasks if t.task_status == "failed"]
    
    status_report.append(f"• Pending: {len(pending_tasks)}")
    status_report.append(f"• In Progress: {len(in_progress_tasks)}")
    status_report.append(f"• Completed: {len(completed_tasks)}")
    status_report.append(f"• Failed: {len(failed_tasks)}")
    
    # Check if all tasks are complete
    all_completed = len(completed_tasks) == len(state.delegated_tasks) and len(state.delegated_tasks) > 0
    if all_completed:
        status_report.append("\n✅ **All tasks are completed!**")
        
        # Check if database operations have already been executed
        if not state.pending_database_operations and not getattr(state, 'database_operations_executed', False):
            status_report.append("\n🚀 **Automatically triggering database operations...**")
            status_message = "\n".join(status_report)
            
            # Return command to automatically collect and execute database operations
            return Command(
                update={
                    "all_tasks_completed": all_completed,
                    "auto_execute_database_ops": True,  # Flag to trigger automatic execution
                    "messages": [ToolMessage(content=status_message, tool_call_id=tool_call_id)],
                }
            )
        elif getattr(state, 'database_operations_executed', False):
            status_report.append("\n🎯 **Database operations completed - widgets are live on dashboard!**")
        else:
            status_report.append("\n📋 **Database operations are pending execution...**")
    
    status_message = "\n".join(status_report)
    
    return Command(
        update={
            "all_tasks_completed": all_completed,
            "messages": [ToolMessage(content=status_message, tool_call_id=tool_call_id)],
        }
    )




@tool
def execute_widget_tasks(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    check_only: bool = False
) -> Command:
    """Execute pending widget tasks OR check task status (consolidated functionality)."""
    try:
        # If check_only is True, just return status report
        if check_only:
            return _generate_task_status_report(state, tool_call_id)
        
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
        
        # NEW: Proper dependency resolution using task_dependencies tracking
        updated_dependencies = []  # Create new list to avoid mutating original
        updated_tasks = state.delegated_tasks.copy()  # Create copy for task updates
        dependencies_resolved = 0
        
        # Process each dependency and create updated versions
        for dependency in state.task_dependencies:
            updated_dependency = TaskDependency(
                task_id=dependency.task_id,
                dependent_on=dependency.dependent_on.copy(),
                reference_widget_ids=dependency.reference_widget_ids.copy()
            )
            
            # Check if we need to resolve dependencies for this task
            pending_task = next(
                (task for task in pending_tasks if task.task_id == dependency.task_id),
                None
            )
            
            if (pending_task and 
                len(updated_dependency.reference_widget_ids) < len(updated_dependency.dependent_on)):
                
                logger.info(f"Resolving dependencies for pending task {dependency.task_id}...")
                
                # Find completed tasks this dependency depends on
                for dependent_task_id in updated_dependency.dependent_on:
                    # Check if this dependent task has completed and has a widget_id
                    completed_task = next(
                        (task for task in state.delegated_tasks 
                         if task.task_id == dependent_task_id and task.task_status == "completed"),
                        None
                    )
                    
                    if completed_task:
                        # Use completed_widget_ids mapping to get the actual widget_id
                        widget_id = state.completed_widget_ids.get(completed_task.task_id)
                        
                        if widget_id and widget_id not in updated_dependency.reference_widget_ids:
                            updated_dependency.reference_widget_ids.append(widget_id)
                            dependencies_resolved += 1
                            logger.info(f"✅ Resolved dependency: Task {dependency.task_id} now has widget {widget_id} from completed task {dependent_task_id}")
            
            updated_dependencies.append(updated_dependency)
        
        # Update the tasks with resolved dependencies
        for task in updated_tasks:
            dependency = next(
                (dep for dep in updated_dependencies if dep.task_id == task.task_id),
                None
            )
            if dependency:
                task.reference_widget_id = dependency.reference_widget_ids.copy()
                logger.info(f"✅ Updated task {task.task_id} reference_widget_id: {task.reference_widget_id}")
        
        if dependencies_resolved > 0:
            logger.info(f"✅ Successfully resolved {dependencies_resolved} task dependencies")
        else:
            logger.info("ℹ️ No new dependencies to resolve at this time")
        
        # Check if we had any pending dependencies to process
        has_pending_dependencies = dependencies_resolved > 0
        
        # Update pending_tasks list after dependency resolution
        all_pending_tasks = [
            task for task in updated_tasks 
            if task.target_agent == "widget_agent_team" and task.task_status == "pending"
        ]
        
        # IMPORTANT: Filter out tasks that have unresolved dependencies
        ready_tasks = []
        blocked_tasks = []
        
        for task in all_pending_tasks:
            # Check if this task has dependencies
            task_dependency = next(
                (dep for dep in updated_dependencies if dep.task_id == task.task_id),
                None
            )
            
            if task_dependency:
                # This task has dependencies - check if they're all resolved
                unresolved_count = len(task_dependency.dependent_on) - len(task_dependency.reference_widget_ids)
                if unresolved_count > 0:
                    blocked_tasks.append(task)
                    logger.info(f"⏳ Task {task.task_id} ({task.widget_type}) blocked - waiting for {unresolved_count} dependencies")
                    continue
                else:
                    logger.info(f"✅ Task {task.task_id} ({task.widget_type}) ready - all dependencies resolved")
            
            ready_tasks.append(task)
        
        if not ready_tasks:
            if blocked_tasks:
                message = f"No tasks ready to execute - {len(blocked_tasks)} tasks blocked waiting for dependencies"
                logger.info(message)
                return Command(
                    update={
                        "task_dependencies": updated_dependencies,
                        "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
                    }
                )
            else:
                message = "No pending widget tasks to execute"
                return Command(
                    update={
                        "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
                    }
                )
        
        # Process the first ready task (prioritizing independent tasks)
        current_task = ready_tasks[0]
        logger.info(f"🎯 Executing ready task: {current_task.task_id} ({current_task.widget_type}) - {len(ready_tasks)} total ready tasks")
        
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
            "reference_widget_id": current_task.reference_widget_id,  # For text blocks referencing other widgets
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
        final_updated_tasks = []
        updated_completed_widget_ids = state.completed_widget_ids.copy()
        
        for task in updated_tasks:
            if task.task_id == current_task.task_id:
                # Update task with result
                task.task_status = widget_result.get("task_status", "completed")
                task.completed_at = datetime.now()
                task.started_at = datetime.now()
                
                logger.info(f"🔍 WIDGET RESULT DEBUG for task {task.task_id}:")
                logger.info(f"  - task_status: {widget_result.get('task_status')}")
                logger.info(f"  - data_validated: {widget_result.get('data_validated')}")
                logger.info(f"  - has_database_operation: {bool(widget_result.get('database_operation'))}")
                logger.info(f"  - has_data: {bool(widget_result.get('data'))}")
                logger.info(f"  - error_messages: {widget_result.get('error_messages', [])}")
                
                if widget_result.get("error_messages"):
                    task.error_message = "; ".join(widget_result["error_messages"])
                    task.task_status = "failed"
                else:
                    # Store the full database operation as the result (not just a message)
                    db_operation = widget_result.get("database_operation")
                    if db_operation:
                        logger.info(f"🎯 STORING DATABASE OPERATION for task {task.task_id}: {db_operation.get('operation_type')} with widget_data keys: {list(db_operation.get('widget_data', {}).keys())}")
                        task.result = db_operation
                        task.database_operation = db_operation
                    else:
                        logger.info(f"❌ NO DATABASE OPERATION found for task {task.task_id} - using fallback string")
                        task.result = f"Widget {current_task.operation} operation completed successfully"
                    
                    # Capture the actual widget_id from the completed task
                    actual_widget_id = widget_result.get("widget_id")
                    if actual_widget_id and task.task_status == "completed":
                        updated_completed_widget_ids[task.task_id] = actual_widget_id
                        logger.info(f"Captured completed widget ID: {actual_widget_id} for task {task.task_id}")
                
                # Update database task status if db_task_id exists
                if task.db_task_id:
                    try:
                        supabase = get_supabase_client()
                        task_update = UpdateTaskInput(
                            task_id=task.db_task_id,
                            status=task.task_status,
                            started_at=task.started_at.isoformat() if task.started_at else None,
                            completed_at=task.completed_at.isoformat() if task.completed_at else None
                        )
                        update_db_task_status(supabase, task_update)
                        logger.info(f"Updated database task {task.db_task_id} status to {task.task_status}")
                    except Exception as db_error:
                        logger.warning(f"Failed to update database task status: {db_error}")
                    
            final_updated_tasks.append(task)
        
        # Create execution message including dependency resolution if it occurred
        execution_message = f"✅ **WIDGET TASK EXECUTED**\n\nCompleted {current_task.operation} {current_task.widget_type} widget task\nStatus: {widget_result.get('task_status', 'completed')}"
        
        # Add dependency resolution message if we resolved any
        if has_pending_dependencies and dependencies_resolved > 0:
            execution_message += f"\n\n🔗 **Resolved {dependencies_resolved} task dependency(ies)** before execution."
        
        logger.info(f"Widget task {current_task.task_id} completed with status: {widget_result.get('task_status')}")
        
        return Command(
            update={
                "delegated_tasks": final_updated_tasks,
                "completed_widget_ids": updated_completed_widget_ids,
                "task_dependencies": updated_dependencies,  # Include updated dependency tracking
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
def finalize_response(
    final_message: str,
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    is_error: bool = False,
    failed_tool: str = None,
    error_messages: List[str] = None
) -> Command:
    """Finalize the response (success or error) - consolidated functionality."""
    try:
        if is_error:
            # Handle error case (consolidated from generate_error_response)
            error_summary = f"I encountered repeated issues while trying to {failed_tool}. " if failed_tool else "I encountered technical difficulties. "
            
            if failed_tool == "analyze_available_data":
                error_summary += "I was unable to access or analyze the data files needed for your request. "
            elif failed_tool == "plan_widget_tasks":
                error_summary += "I had trouble planning the appropriate widgets for your request. "
            elif failed_tool == "execute_widget_tasks":
                error_summary += "I couldn't successfully execute the widget creation tasks. "
            else:
                error_summary += "There was a technical issue with the system. "
            
            # Add specific error details if available
            if error_messages:
                recent_errors = error_messages[-3:]  # Show last 3 errors
                error_details = "\\n".join([f"• {error}" for error in recent_errors])
                error_summary += f"\\n\\nTechnical details:\\n{error_details}"
            
            error_summary += "\\n\\nPlease try your request again later, or contact support if the issue persists."
            
            logger.error(f"Top level supervisor failed: {error_summary}")
            
            return Command(
                goto=END,
                update={
                    "final_response": error_summary,
                    "supervisor_status": "failed",
                    "updated_at": datetime.now(),
                    "messages": [ToolMessage(content=error_summary, tool_call_id=tool_call_id)],
                }
            )
        else:
            # Handle success case
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
        # Fallback error handling
        fallback_message = f"I'm sorry, but I encountered technical difficulties while processing your request. Please try again later or contact support."
        logger.error(f"Error finalizing response: {e}")
        return Command(
            goto=END,
            update={
                "final_response": fallback_message,
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=fallback_message, tool_call_id=tool_call_id)],
            }
        )


# Define the tools available to the supervisor
supervisor_tools = [
    analyze_available_data,    # Data analysis
    plan_widget_tasks,         # Task planning + dependency resolution
    execute_widget_tasks,      # Task execution + status checking  
    finalize_created_widgets,  # Finalize widgets created by widget teams (unified tool)
    finalize_response         # Response finalization + error handling
]

# Create the supervisor agent using create_react_agent with structured output
def create_top_level_supervisor(state: TopLevelSupervisorState, model_name: Optional[str] = None):
    """Create the top-level supervisor agent with Langfuse prompt and model configuration.
    
    This function REQUIRES Langfuse to be available and will fail if prompts or model 
    configuration cannot be fetched from Langfuse. It compiles the prompt with dynamic
    variables from the current state.
    
    Args:
        state: The current supervisor state containing user_prompt and other context
        model_name: Optional model name to override the Langfuse configuration
    
    Raises:
        RuntimeError: If Langfuse is not available or prompts cannot be fetched
        ValueError: If fetched configuration is invalid or incomplete
    """
    
    # Check if Langfuse is available - with better error handling for different execution contexts
    try:
        try:
            from config import LANGFUSE_AVAILABLE
        except ImportError:
            # Handle path issues similar to the prompts import above
            import sys
            import os
            src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from config import LANGFUSE_AVAILABLE
            
        if not LANGFUSE_AVAILABLE:
            # Try direct import to give a more specific error
            try:
                from langfuse import Langfuse
                # Langfuse is actually available, but config says it's not
                logger.warning("Langfuse import works but LANGFUSE_AVAILABLE is False in config")
            except ImportError as langfuse_error:
                raise RuntimeError(
                    f"Langfuse is not available. Cannot create top_level_supervisor without Langfuse integration. "
                    f"Please install langfuse: pip install langfuse. Import error: {langfuse_error}"
                )
    except ImportError as config_error:
        # Try direct langfuse import to see if the issue is with config import or langfuse itself
        try:
            from langfuse import Langfuse
            logger.warning(f"Config import failed even with path adjustment ({config_error}) but langfuse is available directly")
        except ImportError as langfuse_error:
            raise RuntimeError(
                f"Cannot import Langfuse. Please ensure langfuse is installed: pip install langfuse. "
                f"Config error: {config_error}, Langfuse error: {langfuse_error}"
            )
        except Exception as langfuse_error:
            raise RuntimeError(
                f"Config import failed and langfuse has issues. Config error: {config_error}, "
                f"Langfuse error: {langfuse_error}"
            )
    
    try:
        # Prepare runtime variables from current state
        logger.info("Preparing runtime variables for top_level_supervisor prompt compilation...")
        
        # Create a comprehensive state analysis for the prompt
        state_analysis = {
            "supervisor_status": state.supervisor_status,
            "available_data_summary": state.available_data_summary or "No data summary available",
            "available_files_count": len(state.available_files),
            "available_files": state.available_files,
            "delegated_tasks_count": len(state.delegated_tasks),
            "delegated_tasks": [{"title": task.title, "status": task.task_status, "widget_type": task.widget_type} for task in state.delegated_tasks],
            "context_widget_ids": state.context_widget_ids,
            "error_messages": state.error_messages,
            "current_reasoning": state.current_reasoning or "Starting analysis",
            "all_tasks_completed": state.all_tasks_completed,
            "remaining_steps": state.remaining_steps,
            "dashboard_id": state.dashboard_id,
            "chat_id": state.chat_id,
            "user_id": state.user_id,
            "request_id": state.request_id
        }
        
        prompt_variables = {
            "user_prompt": state.user_prompt,
            "current_state_analysis": state_analysis,
        }
        
        # Compile the prompt with dynamic variables from Langfuse (REQUIRED)
        logger.info("Compiling top_level_supervisor prompt from Langfuse with dynamic variables...")
        system_prompt = compile_prompt(
            "top_level_supervisor/top_level_supervisor", 
            prompt_variables,
            label="latest"
        )
        
        # Validate compiled prompt (handle different formats)
        if not system_prompt:
            raise ValueError("Compiled prompt from Langfuse is empty or None")
        
        # Convert to string if needed and validate
        system_prompt_str = str(system_prompt)
        if not system_prompt_str or len(system_prompt_str.strip()) == 0:
            raise ValueError("Compiled prompt from Langfuse is empty or invalid")
        
        # Use the string version for the LLM
        system_prompt = system_prompt_str
        
        logger.info(f"✅ Successfully compiled Langfuse supervisor prompt with {len(prompt_variables)} variables")
            
    except Exception as e:
        error_msg = f"Failed to compile prompt 'top_level_supervisor' from Langfuse: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    try:
        # Fetch model configuration from Langfuse (REQUIRED)
        logger.info("Fetching model configuration from Langfuse...")
        prompt_config = get_prompt_config("top_level_supervisor/top_level_supervisor", label="latest")
        
        # Extract required configuration
        model = prompt_config.get("model")
        temperature = prompt_config.get("temperature")
        reasoning_effort = prompt_config.get("reasoning_effort")
        
        # Validate configuration
        if not model:
            raise ValueError("Model configuration is missing or empty in Langfuse prompt config")
        if temperature is None:
            raise ValueError("Temperature configuration is missing in Langfuse prompt config")
            
        # Override model if explicitly provided
        if model_name:
            logger.info(f"Overriding Langfuse model '{model}' with provided model '{model_name}'")
            model = model_name
            
        logger.info(f"✅ Using Langfuse configuration - model: {model}, temperature: {temperature}, reasoning_effort: {reasoning_effort}")
        
    except Exception as e:
        error_msg = f"Failed to fetch model configuration from Langfuse: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # system_prompt is already validated and converted to string above
    
    try:
        # Initialize the language model with Langfuse configuration
        llm_params = {
            "model": model,
            "temperature": temperature
        }
        
        # Add reasoning_effort if provided (for reasoning models like o1, o3, o4-mini)
        if reasoning_effort:
            llm_params["reasoning_effort"] = reasoning_effort
            
        llm = ChatOpenAI(**llm_params)
        
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
            
        # Check if we should automatically execute database operations
        if getattr(supervisor_state, 'auto_execute_database_ops', False):
            try:
                logger.info("🚀 Auto-finalizing created widgets...")
                
                # Finalize widgets created by completed widget agent tasks
                finalize_result = finalize_created_widgets(supervisor_state, "auto_finalize")
                
                # Update state to mark widgets as finalized
                if hasattr(finalize_result, 'update'):
                    for key, value in finalize_result.update.items():
                        setattr(supervisor_state, key, value)
                    # Mark widgets as finalized
                    supervisor_state.database_operations_executed = True
                    supervisor_state.auto_execute_database_ops = False  # Clear the flag
                
                logger.info("✅ Auto-execution of database operations completed")
                
            except Exception as e:
                logger.error(f"Failed to auto-execute database operations: {e}")
                # Continue with normal supervisor flow even if auto-execution fails
            
        # Create the supervisor agent with current state for dynamic variable compilation
        supervisor_agent = create_top_level_supervisor(supervisor_state)
        
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
            # Create Langfuse callback handler for tracing
            langfuse_config = {}
            if LANGFUSE_AVAILABLE:
                try:
                    langfuse_handler = get_langfuse_callback_handler(
                        trace_name="top-level-supervisor-execution",
                        session_id=supervisor_state.chat_id,
                        user_id=supervisor_state.user_id,
                        tags=["supervisor", "orchestrator", "top-level"],
                        metadata={
                            "dashboard_id": supervisor_state.dashboard_id,
                            "request_id": supervisor_state.request_id,
                            "user_prompt": supervisor_state.user_prompt[:200] + "..." if len(supervisor_state.user_prompt) > 200 else supervisor_state.user_prompt
                        }
                    )
                    if langfuse_handler:
                        langfuse_config = {"callbacks": [langfuse_handler]}
                        logger.info("Top Level Supervisor configured with Langfuse tracing")
                except Exception as langfuse_error:
                    logger.warning(f"Failed to create Langfuse handler for supervisor: {langfuse_error}")
            
            # Invoke the agent with or without tracing
            if langfuse_config:
                result = supervisor_agent.invoke(agent_input, config=langfuse_config)
            else:
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