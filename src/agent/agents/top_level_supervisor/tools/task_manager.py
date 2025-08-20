"""Task management tools for the Top Level Supervisor."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from agent.models import DelegatedTask, TopLevelSupervisorState

logger = logging.getLogger(__name__)


def create_task(
    state: TopLevelSupervisorState,
    target_agent: str,
    task_instructions: str,
    widget_type: str,
    operation: str,
    file_ids: List[str],
    widget_id: Optional[str] = None,
    widget_agent_state_data: Optional[Dict[str, Any]] = None
) -> TopLevelSupervisorState:
    """Create a new delegated task and add it to the state.
    
    Args:
        state: Current supervisor state
        target_agent: Which agent team should handle this (widget_agent_team)
        task_instructions: Detailed instructions for the task
        widget_type: Type of widget to create/update/delete
        operation: Widget operation (CREATE, UPDATE, DELETE)
        file_ids: File IDs to use for the widget
        widget_id: Widget ID for UPDATE/DELETE operations or context reference
        widget_agent_state_data: Data needed to initialize WidgetAgentState
        
    Returns:
        Updated state with the new task added
    """
    try:
        new_task = DelegatedTask(
            target_agent=target_agent,
            task_instructions=task_instructions,
            widget_type=widget_type,
            operation=operation,
            file_ids=file_ids,
            widget_id=widget_id,
            widget_agent_state_data=widget_agent_state_data,
            created_at=datetime.now()
        )
        
        # Add task to the delegated tasks list
        updated_tasks = state.delegated_tasks.copy()
        updated_tasks.append(new_task)
        
        # Update state
        state.delegated_tasks = updated_tasks
        state.updated_at = datetime.now()
        
        logger.info(f"Created new task {new_task.task_id} for {target_agent}: {operation} {widget_type} widget")
        
        return state
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        state.error_messages.append(f"Failed to create task: {str(e)}")
        return state


def update_task_status(
    state: TopLevelSupervisorState,
    task_id: str,
    new_status: str,
    result: Optional[str] = None,
    error_message: Optional[str] = None
) -> TopLevelSupervisorState:
    """Update the status of a delegated task.
    
    Args:
        state: Current supervisor state
        task_id: ID of the task to update
        new_status: New status (pending, in_progress, completed, failed)
        result: Task result summary message if completed
        error_message: Error message if failed
        
    Returns:
        Updated state with task status changed
    """
    try:
        # Find the task to update
        updated_tasks = []
        task_found = False
        
        for task in state.delegated_tasks:
            if task.task_id == task_id:
                task_found = True
                # Update task status and timestamps
                task.task_status = new_status
                
                if new_status == "in_progress" and not task.started_at:
                    task.started_at = datetime.now()
                elif new_status in ["completed", "failed"]:
                    task.completed_at = datetime.now()
                    
                if result:
                    task.result = result
                if error_message:
                    task.error_message = error_message
                    
            updated_tasks.append(task)
        
        if not task_found:
            logger.warning(f"Task {task_id} not found for status update")
            state.error_messages.append(f"Task {task_id} not found")
            return state
        
        # Update state
        state.delegated_tasks = updated_tasks
        state.updated_at = datetime.now()
        
        # Check if all tasks are completed
        state.all_tasks_completed = all(
            task.task_status in ["completed", "failed"] 
            for task in state.delegated_tasks
        )
        
        logger.info(f"Updated task {task_id} status to {new_status}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error updating task status: {e}")
        state.error_messages.append(f"Failed to update task status: {str(e)}")
        return state


def get_pending_tasks(state: TopLevelSupervisorState) -> list[DelegatedTask]:
    """Get all pending tasks from the state.
    
    Args:
        state: Current supervisor state
        
    Returns:
        List of pending tasks
    """
    return [task for task in state.delegated_tasks if task.task_status == "pending"]


def get_completed_tasks(state: TopLevelSupervisorState) -> list[DelegatedTask]:
    """Get all completed tasks from the state.
    
    Args:
        state: Current supervisor state
        
    Returns:
        List of completed tasks
    """
    return [task for task in state.delegated_tasks if task.task_status == "completed"]


def get_failed_tasks(state: TopLevelSupervisorState) -> list[DelegatedTask]:
    """Get all failed tasks from the state.
    
    Args:
        state: Current supervisor state
        
    Returns:
        List of failed tasks
    """
    return [task for task in state.delegated_tasks if task.task_status == "failed"]


def delegate_to_widget_team(state: TopLevelSupervisorState) -> str:
    """Delegate a pending widget task to the widget_agent_team.
    
    This function identifies the next pending widget task and prepares
    for delegation to the widget_agent_team node.
    
    Args:
        state: Current supervisor state
        
    Returns:
        "widget_agent_team" to route to the widget team node, or status message
    """
    try:
        # Find pending widget tasks
        pending_widget_tasks = [
            task for task in state.delegated_tasks 
            if task.target_agent == "widget_agent_team" and task.task_status == "pending"
        ]
        
        if not pending_widget_tasks:
            logger.info("No pending widget tasks found for delegation")
            return "No pending widget tasks to delegate"
        
        # Mark the first task as in_progress
        current_task = pending_widget_tasks[0]
        current_task.task_status = "in_progress"
        current_task.started_at = datetime.now()
        
        state.current_reasoning = f"Delegating widget task: {current_task.task_instructions}"
        state.updated_at = datetime.now()
        
        logger.info(f"Delegating task {current_task.task_id} to widget_agent_team")
        
        # Return the node name to route to
        return "widget_agent_team"
        
    except Exception as e:
        error_msg = f"Error delegating to widget team: {str(e)}"
        logger.error(error_msg)
        state.error_messages.append(error_msg)
        return f"Delegation failed: {str(e)}"