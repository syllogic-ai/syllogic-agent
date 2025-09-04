"""Helper functions for LangGraph operations.

This module contains business logic that was moved from graph.py to follow
CLAUDE.md guidelines. The graph.py should only contain graph structure definition.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List

from langgraph.types import Command

logger = logging.getLogger(__name__)


def should_delegate_to_widget_team(state):
    """Conditional edge function to determine if we should delegate to widget team.
    
    This function checks the supervisor's reasoning and state to determine
    if tasks should be delegated to the widget_agent_team.
    """
    # Handle both dict and object state formats
    def get_state_value(state, key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        else:
            return getattr(state, key, default)
    
    # Check if supervisor is completed or failed FIRST (highest priority)
    supervisor_status = get_state_value(state, 'supervisor_status', 'analyzing')
    if supervisor_status in ["completed", "failed"]:
        return "__end__"
    
    # Check if supervisor wants to delegate tasks (look for the delegation signal)
    current_reasoning = get_state_value(state, 'current_reasoning', '')
    if current_reasoning and "DELEGATE_TO_WIDGET_TEAM" in current_reasoning:
        logger.info(f"Delegating to widget team based on reasoning: {current_reasoning}")
        return "widget_agent_team"
    
    # Check if there are pending widget tasks that need processing
    delegated_tasks = get_state_value(state, 'delegated_tasks', [])
    pending_widget_tasks = [
        task for task in delegated_tasks 
        if (isinstance(task, dict) and task.get("target_agent") == "widget_agent_team" and task.get("task_status") == "pending")
        or (hasattr(task, 'target_agent') and task.target_agent == "widget_agent_team" and task.task_status == "pending")
    ]
    
    if pending_widget_tasks:
        logger.info(f"Found {len(pending_widget_tasks)} pending widget tasks - delegating to widget team")
        return "widget_agent_team"
    
    # Continue with supervisor
    return "top_level_supervisor"


def convert_backend_payload_to_graph_state(payload: dict) -> dict:
    """Convert BackendPayload format to SupervisorGraphState format.
    
    This allows the LangGraph API to accept BackendPayload format
    and automatically convert it to the expected graph state.
    """
    from agent.models import BackendPayload
    
    # Validate as BackendPayload first
    validated_payload = BackendPayload(**payload)
    
    # Convert to SupervisorGraphState format
    return {
        "messages": [],
        "remaining_steps": 10,
        "user_prompt": validated_payload.message,  # message -> user_prompt mapping
        "user_id": validated_payload.user_id,
        "dashboard_id": validated_payload.dashboard_id,
        "chat_id": validated_payload.chat_id,
        "request_id": validated_payload.request_id,
        "file_ids": validated_payload.file_ids,
        "context_widget_ids": validated_payload.context_widget_ids or [],
        "available_files": [],
        "available_data_summary": None,
        "delegated_tasks": [],
        "supervisor_status": "analyzing",
        "current_reasoning": None,
        "final_response": None,
        "error_messages": [],
        "created_at": datetime.now(),
        "updated_at": None,
        "all_tasks_completed": False
    }


def widget_team_adapter(state: dict):
    """Adapter function to handle widget team delegation with proper state transformation."""
    try:
        logger.info("Widget team adapter called - processing delegation")
        
        # Import here to avoid circular dependencies
        from config import create_langfuse_config
        
        # Get the first pending widget task
        delegated_tasks = state.get("delegated_tasks", [])
        pending_task = None
        for task in delegated_tasks:
            task_data = task if isinstance(task, dict) else task.__dict__
            if task_data.get("target_agent") == "widget_agent_team" and task_data.get("task_status") == "pending":
                pending_task = task_data
                break
        
        if not pending_task:
            logger.warning("No pending widget task found for delegation")
            # Check if there's widget team input from the delegation
            widget_input = state.get("widget_team_input") or (state.get("widget_team_inputs", [{}])[0])
            if widget_input:
                pending_task = widget_input
            else:
                return Command(update={
                    "current_reasoning": "No pending widget task found",
                    "supervisor_status": "analyzing",
                    # Preserve required fields that must not be lost
                    "user_prompt": state.get("user_prompt", ""),
                    "dashboard_id": state.get("dashboard_id", ""),
                    "user_id": state.get("user_id", f"user_{uuid.uuid4().hex[:8]}"),
                    "chat_id": state.get("chat_id", f"chat_{uuid.uuid4().hex[:8]}"),
                    "request_id": state.get("request_id", f"req_{uuid.uuid4().hex[:8]}"),
                })
        
        # Transform the task data to WidgetAgentState format
        widget_state = {
            "messages": state.get("messages", []),
            "remaining_steps": state.get("remaining_steps", 10),
            "task_id": pending_task.get("task_id", str(uuid.uuid4())),
            "task_instructions": pending_task.get("task_instructions", ""),
            "user_prompt": pending_task.get("user_prompt", state.get("user_prompt", "")),
            "operation": pending_task.get("operation", "CREATE"),
            "widget_type": pending_task.get("widget_type", "bar"),
            "dashboard_id": pending_task.get("dashboard_id", state.get("dashboard_id", "")),
            "chat_id": pending_task.get("chat_id", state.get("chat_id", "")),
            "file_ids": pending_task.get("file_ids", []),
            "widget_id": pending_task.get("widget_id"),
            "title": pending_task.get("title", "Generated Widget"),
            "description": pending_task.get("description", "Widget generated by AI"),
            "task_status": "in_progress",  # Mark as in progress when delegating
            "created_at": datetime.now(),
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
        
        logger.info(f"Invoking widget subgraph for {widget_state['operation']} {widget_state['widget_type']} widget")
        
        # Build and invoke the widget subgraph with Langfuse tracing
        from agent.graph import build_widget_agent_graph
        widget_graph = build_widget_agent_graph()
        langfuse_config = create_langfuse_config(state, "widget-subgraph-execution")
        
        if langfuse_config:
            # Invoke with Langfuse tracing
            result = widget_graph.invoke(widget_state, config=langfuse_config)
        else:
            # Invoke without tracing
            result = widget_graph.invoke(widget_state)
        
        # Update the delegated task status in parent state
        updated_tasks = []
        for task in delegated_tasks:
            task_data = task if isinstance(task, dict) else task.__dict__
            if task_data.get("task_id") == pending_task.get("task_id"):
                # Update task status based on widget result
                task_data["task_status"] = result.get("task_status", "completed")
                task_data["completed_at"] = datetime.now()
                if result.get("error_messages"):
                    task_data["error_message"] = "; ".join(result["error_messages"])
                else:
                    # Store the full database operation as the result (not just a message)
                    if result.get("database_operation"):
                        task_data["result"] = result.get("database_operation")
                        task_data["database_operation"] = result.get("database_operation")
                    else:
                        task_data["result"] = "Widget operation completed successfully"
            updated_tasks.append(task)
        
        logger.info(f"Widget team completed task with status: {result.get('task_status')}")
        
        # Return Command to update parent state - preserve required fields
        return Command(update={
            "delegated_tasks": updated_tasks,
            "current_reasoning": f"Widget task completed: {result.get('task_status')}",
            "supervisor_status": "analyzing",  # Return to analyzing to check for more tasks
            "updated_at": datetime.now(),
            # Preserve required fields that must not be lost
            "user_prompt": state.get("user_prompt", ""),
            "dashboard_id": state.get("dashboard_id", ""),
            "user_id": state.get("user_id", f"user_{uuid.uuid4().hex[:8]}"),
            "chat_id": state.get("chat_id", f"chat_{uuid.uuid4().hex[:8]}"),
            "request_id": state.get("request_id", f"req_{uuid.uuid4().hex[:8]}"),
        })
        
    except Exception as e:
        logger.error(f"Widget team adapter error: {e}")
        return Command(update={
            "error_messages": state.get("error_messages", []) + [f"Widget team error: {str(e)}"],
            "current_reasoning": f"Widget team failed: {str(e)}",
            "supervisor_status": "failed",
            # Preserve required fields that must not be lost
            "user_prompt": state.get("user_prompt", ""),
            "dashboard_id": state.get("dashboard_id", ""),
            "user_id": state.get("user_id", f"user_{uuid.uuid4().hex[:8]}"),
            "chat_id": state.get("chat_id", f"chat_{uuid.uuid4().hex[:8]}"),
            "request_id": state.get("request_id", f"req_{uuid.uuid4().hex[:8]}"),
        })