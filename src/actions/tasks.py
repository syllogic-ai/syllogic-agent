"""Task management utilities for Supabase tasks table.
Provides functions to create, update, and manage tasks.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from supabase import Client

from agent.models import Task, CreateTaskInput, UpdateTaskInput, DelegatedTask

logger = logging.getLogger(__name__)


def create_task(supabase: Client, task_input: CreateTaskInput) -> Task:
    """Create a new task in the database.

    Args:
        supabase: Supabase client instance
        task_input: CreateTaskInput containing task data

    Returns:
        Task model with the created task data

    Raises:
        Exception: If task creation fails
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        # Create task data
        task_data = {
            "id": task_id,
            "chat_id": task_input.chat_id,
            "dashboard_id": task_input.dashboard_id,
            "task_group_id": task_input.task_group_id,
            "title": task_input.title,
            "description": task_input.description,
            "status": task_input.status,
            "order": task_input.order,
            "created_at": current_time,
            "updated_at": current_time,
        }

        # Insert task into database
        result = supabase.table("tasks").insert(task_data).execute()

        if result.data:
            logger.info(f"Created task {task_id}: {task_input.title}")
            return Task(**result.data[0])
        else:
            raise Exception(f"Failed to create task: No data returned")

    except Exception as e:
        logger.error(f"Error creating task '{task_input.title}': {str(e)}")
        raise


def create_tasks_from_delegated_tasks(
    supabase: Client, 
    delegated_tasks: List[DelegatedTask], 
    chat_id: str, 
    dashboard_id: str, 
    task_group_id: str
) -> List[Task]:
    """Create database tasks from a list of DelegatedTask objects.

    Args:
        supabase: Supabase client instance
        delegated_tasks: List of DelegatedTask objects to convert
        chat_id: Chat ID to associate with tasks
        dashboard_id: Dashboard ID to associate with tasks
        task_group_id: Task group ID to link tasks together

    Returns:
        List of created Task models

    Raises:
        Exception: If task creation fails
    """
    try:
        created_tasks = []
        current_time = datetime.now().isoformat()

        for i, delegated_task in enumerate(delegated_tasks):
            # Create task input
            task_input = CreateTaskInput(
                chat_id=chat_id,
                dashboard_id=dashboard_id,
                task_group_id=task_group_id,
                title=delegated_task.task_title,
                description=f"{delegated_task.operation} {delegated_task.widget_type} widget: {delegated_task.description}",
                status="pending",
                order=i + 1
            )

            # Create the task
            task = create_task(supabase, task_input)
            
            # Update the delegated task with database info
            delegated_task.db_task_id = task.id
            delegated_task.task_group_id = task_group_id
            
            created_tasks.append(task)

        logger.info(f"Created {len(created_tasks)} database tasks from delegated tasks")
        return created_tasks

    except Exception as e:
        logger.error(f"Error creating tasks from delegated tasks: {str(e)}")
        raise


def update_task_status(supabase: Client, task_update: UpdateTaskInput) -> Task:
    """Update task status and timestamps.

    Args:
        supabase: Supabase client instance
        task_update: UpdateTaskInput containing update data

    Returns:
        Updated Task model

    Raises:
        Exception: If task update fails
    """
    try:
        update_data = {
            "updated_at": datetime.now().isoformat()
        }

        # Add optional fields if provided
        if task_update.status:
            update_data["status"] = task_update.status
        if task_update.started_at:
            update_data["started_at"] = task_update.started_at
        if task_update.completed_at:
            update_data["completed_at"] = task_update.completed_at

        # Update task in database
        result = (
            supabase.table("tasks")
            .update(update_data)
            .eq("id", task_update.task_id)
            .execute()
        )

        if result.data:
            logger.info(f"Updated task {task_update.task_id} status to {task_update.status}")
            return Task(**result.data[0])
        else:
            raise Exception(f"Failed to update task {task_update.task_id}: No data returned")

    except Exception as e:
        logger.error(f"Error updating task {task_update.task_id}: {str(e)}")
        raise


def get_tasks_by_group(supabase: Client, task_group_id: str) -> List[Task]:
    """Get all tasks belonging to a specific task group.

    Args:
        supabase: Supabase client instance
        task_group_id: Task group ID to filter by

    Returns:
        List of Task models ordered by order field

    Raises:
        Exception: If retrieval fails
    """
    try:
        result = (
            supabase.table("tasks")
            .select("*")
            .eq("task_group_id", task_group_id)
            .order("order", desc=False)
            .execute()
        )

        tasks = [Task(**task_data) for task_data in result.data]
        logger.info(f"Retrieved {len(tasks)} tasks for group {task_group_id}")
        return tasks

    except Exception as e:
        logger.error(f"Error retrieving tasks for group {task_group_id}: {str(e)}")
        raise


def get_tasks_by_chat(supabase: Client, chat_id: str) -> List[Task]:
    """Get all tasks belonging to a specific chat.

    Args:
        supabase: Supabase client instance
        chat_id: Chat ID to filter by

    Returns:
        List of Task models ordered by creation time

    Raises:
        Exception: If retrieval fails
    """
    try:
        result = (
            supabase.table("tasks")
            .select("*")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
            .execute()
        )

        tasks = [Task(**task_data) for task_data in result.data]
        logger.info(f"Retrieved {len(tasks)} tasks for chat {chat_id}")
        return tasks

    except Exception as e:
        logger.error(f"Error retrieving tasks for chat {chat_id}: {str(e)}")
        raise


def generate_task_group_id(request_id: str) -> str:
    """Generate a task group ID from request ID.

    Args:
        request_id: Request ID to base the group ID on

    Returns:
        Task group ID string
    """
    return f"group_{request_id}_{uuid.uuid4().hex[:8]}"


def format_task_list_message(tasks: List[Task], task_group_id: str) -> str:
    """Format a list of tasks into a user-friendly message.

    Args:
        tasks: List of Task objects to format
        task_group_id: Task group ID for reference

    Returns:
        Formatted task list message
    """
    if not tasks:
        return "No tasks found."

    message_lines = [
        f"📋 **Task List** (Group: {task_group_id[-8:]})",  # Show last 8 chars of group ID
        ""
    ]

    for task in sorted(tasks, key=lambda t: t.order):
        status_emoji = {
            "pending": "⏳",
            "in-progress": "🔄",
            "completed": "✅",
            "failed": "❌"
        }.get(task.status, "❓")

        message_lines.append(f"{task.order}. {status_emoji} **{task.title}** - {task.status}")
        if task.description:
            message_lines.append(f"   _{task.description}_")
        message_lines.append("")

    return "\n".join(message_lines)