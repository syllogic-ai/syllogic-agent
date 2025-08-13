"""Task management tools for the top level coordinator.
Handles initialization, updates, and tracking of widget tasks.
"""

import logging
import uuid
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ...models import TopLevelState, WidgetTask

logger = logging.getLogger(__name__)


async def initialize_tasks(
    state: TopLevelState, llm: ChatOpenAI = None
) -> TopLevelState:
    """Initialize widget tasks based on user request and available data.

    Args:
        state: Current chart generation state
        llm: Optional LLM instance

    Returns:
        Updated state with initialized tasks
    """
    try:
        logger.info("Initializing widget tasks")

        if not llm:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # Prepare context for task generation
        data_context = _prepare_task_generation_context(state)

        system_prompt = """
        You are a dashboard task generator. Based on the user's request and available data, create specific widget tasks.
        
        Each task should be a JSON object with:
        {
            "operation": "CREATE" | "UPDATE" | "DELETE",
            "widget_type": "line" | "bar" | "pie" | "kpi" | "table" | "scatter",
            "title": "Clear, descriptive title",
            "description": "Detailed description of the widget",
            "data_requirements": ["list", "of", "column", "names", "needed"],
            "file_id": "which file to use",
            "priority": 1-5 (1 = highest priority),
            "config_suggestions": {
                "x_axis": "column_name",
                "y_axis": "column_name", 
                "group_by": "optional_column_name",
                "aggregation": "sum|avg|count|max|min"
            }
        }
        
        Return a JSON array of task objects. Create 1-4 tasks based on the request complexity.
        """

        user_prompt = f"""
        USER REQUEST: {state.user_prompt}
        
        AVAILABLE DATA CONTEXT:
        {data_context}
        
        Generate appropriate widget tasks to fulfill this request.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)

        # Parse response
        import json

        try:
            task_data = json.loads(response.content)
            if not isinstance(task_data, list):
                task_data = [task_data]
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response for task generation")
            task_data = []

        # Convert to WidgetTask objects
        widget_tasks = []
        for task_info in task_data:
            try:
                task = WidgetTask(
                    task_id=str(uuid.uuid4()),
                    operation=task_info.get("operation", "CREATE"),
                    widget_type=task_info.get("widget_type", "line"),
                    title=task_info.get("title", "Generated Widget"),
                    description=task_info.get("description", ""),
                    data_requirements=task_info.get("data_requirements", []),
                    file_id=task_info.get(
                        "file_id", state.file_ids[0] if state.file_ids else ""
                    ),
                    priority=task_info.get("priority", 3),
                    config_suggestions=task_info.get("config_suggestions", {}),
                )
                widget_tasks.append(task)
                logger.info(f"Created task: {task.title} ({task.widget_type})")
            except Exception as e:
                logger.error(f"Error creating task from LLM data: {str(e)}")

        # Update state
        state.widget_tasks = widget_tasks

        # Initialize task tracking
        task_status = {}
        for task in widget_tasks:
            task_status[task.task_id] = "pending"

        state.task_completion_status = task_status

        logger.info(f"Initialized {len(widget_tasks)} tasks successfully")
        return state

    except Exception as e:
        error_msg = f"Error initializing tasks: {str(e)}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        return state


async def update_task_status(
    state: TopLevelState, task_id: str, new_status: str, error_message: str = None
) -> TopLevelState:
    """Update the status of a specific task.

    Args:
        state: Current chart generation state
        task_id: Task identifier
        new_status: New status ("pending", "in_progress", "completed", "failed")
        error_message: Optional error message if status is "failed"

    Returns:
        Updated state
    """
    try:
        if task_id not in state.task_completion_status:
            logger.warning(f"Task {task_id} not found in completion status")
            return state

        old_status = state.task_completion_status[task_id]
        state.task_completion_status[task_id] = new_status

        # Find the task and update it if needed
        for task in state.widget_tasks:
            if task.task_id == task_id:
                if error_message and new_status == "failed":
                    # Store error in task description or create an error field
                    task.description = f"{task.description}\n\nError: {error_message}"
                break

        logger.info(f"Updated task {task_id} status: {old_status} -> {new_status}")

        # Add to warnings or errors as appropriate
        if new_status == "completed":
            task_title = next(
                (task.title for task in state.widget_tasks if task.task_id == task_id),
                task_id,
            )
            state.warnings.append(f"Completed task: {task_title}")
        elif new_status == "failed" and error_message:
            state.errors.append(f"Task failed ({task_id}): {error_message}")

        return state

    except Exception as e:
        error_msg = f"Error updating task status: {str(e)}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        return state


def get_next_task(state: TopLevelState) -> WidgetTask:
    """Get the next task to process based on priority and status.

    Args:
        state: Current chart generation state

    Returns:
        Next task to process, or None if no tasks available
    """
    try:
        # Get pending tasks sorted by priority
        pending_tasks = []
        for task in state.widget_tasks:
            if state.task_completion_status.get(task.task_id) == "pending":
                pending_tasks.append(task)

        if not pending_tasks:
            logger.info("No pending tasks available")
            return None

        # Sort by priority (1 = highest priority)
        pending_tasks.sort(key=lambda t: t.priority)

        next_task = pending_tasks[0]
        logger.info(
            f"Selected next task: {next_task.title} (priority {next_task.priority})"
        )

        return next_task

    except Exception as e:
        logger.error(f"Error getting next task: {str(e)}")
        return None


def get_task_summary(state: TopLevelState) -> Dict[str, Any]:
    """Get a summary of all tasks and their status.

    Args:
        state: Current chart generation state

    Returns:
        Dict with task summary information
    """
    try:
        summary = {
            "total_tasks": len(state.widget_tasks),
            "status_counts": {},
            "tasks_by_priority": {},
            "completed_tasks": [],
            "failed_tasks": [],
            "pending_tasks": [],
        }

        # Count statuses
        for task_id, status in state.task_completion_status.items():
            summary["status_counts"][status] = (
                summary["status_counts"].get(status, 0) + 1
            )

        # Group by priority and status
        for task in state.widget_tasks:
            priority = task.priority
            status = state.task_completion_status.get(task.task_id, "unknown")

            if priority not in summary["tasks_by_priority"]:
                summary["tasks_by_priority"][priority] = {
                    "pending": 0,
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                }

            summary["tasks_by_priority"][priority][status] = (
                summary["tasks_by_priority"][priority].get(status, 0) + 1
            )

            # Add to appropriate lists
            if status == "completed":
                summary["completed_tasks"].append(
                    {"id": task.task_id, "title": task.title}
                )
            elif status == "failed":
                summary["failed_tasks"].append(
                    {"id": task.task_id, "title": task.title}
                )
            elif status == "pending":
                summary["pending_tasks"].append(
                    {"id": task.task_id, "title": task.title, "priority": task.priority}
                )

        return summary

    except Exception as e:
        logger.error(f"Error creating task summary: {str(e)}")
        return {"error": str(e)}


def _prepare_task_generation_context(state: TopLevelState) -> str:
    """Prepare context information for task generation.

    Args:
        state: Current chart generation state

    Returns:
        String with context information
    """
    try:
        context_parts = []

        # File information
        if state.file_ids:
            context_parts.append(f"Available files: {', '.join(state.file_ids)}")

        # Schema information (summarized)
        if state.available_data_schemas:
            context_parts.append(
                f"\nData schemas available for {len(state.available_data_schemas)} files:"
            )
            for file_id, schema in list(state.available_data_schemas.items())[
                :3
            ]:  # Limit to 3 files
                column_names = [
                    col.name for col in schema.columns[:10]
                ]  # First 10 columns
                context_parts.append(f"  {file_id}: {', '.join(column_names)}")
                if len(schema.columns) > 10:
                    context_parts.append(
                        f"    ... and {len(schema.columns) - 10} more columns"
                    )

        # Sample data context
        if state.available_sample_data:
            context_parts.append(
                f"\nSample data available for {len(state.available_sample_data)} files"
            )

        return (
            "\n".join(context_parts) if context_parts else "No data context available"
        )

    except Exception as e:
        logger.error(f"Error preparing task generation context: {str(e)}")
        return f"Error preparing context: {str(e)}"
