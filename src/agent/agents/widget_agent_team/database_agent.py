"""Database agent for widget operations."""

import logging
import uuid
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent.models import WidgetAgentState, CreateWidgetInput, UpdateWidgetInput
from .tools.widget_summary import generate_widget_summary

# Initialize logger
logger = logging.getLogger(__name__)

# Handle imports for different execution contexts
try:
    from actions.dashboard import create_widget, update_widget, delete_widget
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.dashboard import create_widget, update_widget, delete_widget


class DatabaseAgent:
    """Agent for handling database operations for widgets."""

    def __init__(self):
        """Initialize database agent."""
        pass

    def execute_database_operations(self, state: WidgetAgentState) -> Command:
        """
        Execute database operations directly and return confirmed widget_id.
        Returns only the widget_id to the top-level supervisor.
        """
        try:
            # Extract operation type
            operation = state.operation
            
            # Extract title - for DELETE operations, use state.title directly
            if operation == "DELETE":
                title = state.title
            else:
                # For CREATE/UPDATE, try to get title from widget_config if available
                title = state.widget_config.get("title", state.title) if state.widget_config else state.title
            
            # Use widget_type from state
            widget_type = state.widget_type
            
            # Use dashboard_id from state
            dashboard_id = state.dashboard_id

            if operation == "CREATE":
                # Generate widget_id as "wid_" + uuid() if not already provided
                widget_id = state.widget_id
                if not widget_id:
                    widget_id = f"wid_{str(uuid.uuid4())}"
                
                # Generate widget summary using Langfuse
                try:
                    widget_summary = generate_widget_summary(
                        widget_type=widget_type,
                        title=title,
                        description=state.description or "",
                        widget_config=state.widget_config or {},
                        tool_call_id=f"summary_gen_{widget_id}"
                    )
                except Exception as e:
                    # Log but don't fail the operation if summary generation fails
                    widget_summary = f"Widget of type {widget_type} titled '{title}'"
                    print(f"Warning: Summary generation failed: {e}")
                
                # Create widget input object with is_configured=False
                create_input = CreateWidgetInput(
                    dashboard_id=dashboard_id,
                    title=title,
                    widget_type=widget_type,
                    config=state.widget_config or {},  # Use unified widget_config
                    widget_id=widget_id,  # Pass the widget_id to be used
                    description=state.description,
                    summary=widget_summary,  # Add the generated summary
                    data={"data": state.widget_config.get("data", []) if state.widget_config else []},  # Extract data from widget_config
                    is_configured=False,  # Explicitly set to False
                )
                
                # Execute database creation immediately
                created_widget = create_widget(create_input)
                
                # Verify the widget was created successfully
                confirmed_widget_id = created_widget.id
                if confirmed_widget_id != widget_id:
                    logger.warning(f"Widget created with different ID: expected {widget_id}, got {confirmed_widget_id}")
                
                logger.info(f"Successfully created widget {confirmed_widget_id} in database")
                
                success_msg = f"✅ WIDGET CREATED: Successfully created widget '{title}' with confirmed ID: {confirmed_widget_id}"
                return Command(
                    goto="END",
                    update={
                        "widget_id": confirmed_widget_id,  # Return only the confirmed widget_id
                        "updated_at": datetime.now(),
                        "task_status": "completed",
                        "widget_creation_completed": True,
                        "messages": [
                            ToolMessage(
                                content=success_msg,
                                tool_call_id="db_create_complete",
                            )
                        ],
                    },
                )
                
            elif operation == "UPDATE":
                # Generate updated widget summary using Langfuse
                try:
                    widget_summary = generate_widget_summary(
                        widget_type=widget_type,
                        title=title,
                        description=state.description or "",
                        widget_config=state.widget_config or {},
                        tool_call_id=f"summary_update_{state.widget_id}"
                    )
                except Exception as e:
                    # Log but don't fail the operation if summary generation fails
                    widget_summary = f"Updated widget of type {widget_type} titled '{title}'"
                    print(f"Warning: Summary generation failed for update: {e}")
                
                # Update existing widget using unified widget_config with is_configured=False
                update_input = UpdateWidgetInput(
                    widget_id=state.widget_id,
                    title=title,
                    widget_type=widget_type,
                    config=state.widget_config or {},  # Use unified widget_config
                    description=state.description,
                    summary=widget_summary,  # Add the updated summary
                    data={"data": state.widget_config.get("data", []) if state.widget_config else []},  # Extract data from widget_config
                    is_configured=False,  # Explicitly set to False
                )
                
                # Execute database update immediately
                updated_widget = update_widget(update_input)
                
                logger.info(f"Successfully updated widget {updated_widget.id} in database")
                
                success_msg = f"✅ WIDGET UPDATED: Successfully updated widget '{title}' with ID: {updated_widget.id}"
                return Command(
                    goto="END",
                    update={
                        "widget_id": updated_widget.id,  # Return only the confirmed widget_id
                        "updated_at": datetime.now(),
                        "task_status": "completed",
                        "widget_update_completed": True,
                        "messages": [
                            ToolMessage(
                                content=success_msg,
                                tool_call_id="db_update_complete",
                            )
                        ],
                    },
                )
                
            elif operation == "DELETE":
                # Execute database deletion immediately
                deletion_success = delete_widget(state.widget_id)
                
                if deletion_success:
                    logger.info(f"Successfully deleted widget {state.widget_id} from database")
                    success_msg = f"✅ WIDGET DELETED: Successfully deleted widget '{title}' with ID: {state.widget_id}"
                    return Command(
                        goto="END",
                        update={
                            "widget_id": state.widget_id,  # Return the widget_id that was deleted
                            "updated_at": datetime.now(),
                            "task_status": "completed",
                            "widget_deletion_completed": True,
                            "messages": [
                                ToolMessage(
                                    content=success_msg,
                                    tool_call_id="db_delete_complete",
                                )
                            ],
                        },
                    )
                else:
                    error_msg = f"Widget {state.widget_id} was not found for deletion"
                    logger.warning(error_msg)
                    return Command(
                        goto="widget_supervisor",
                        update={
                            "error_messages": state.error_messages + [error_msg],
                            "task_status": "failed",
                            "updated_at": datetime.now(),
                        },
                    )
            else:
                # Invalid operation
                error_msg = f"Invalid operation '{operation}'. Must be CREATE, UPDATE, or DELETE"
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "task_status": "failed",
                        "updated_at": datetime.now(),
                    },
                )

        except Exception as e:
            error_msg = f"Database operation execution error: {str(e)}"
            logger.error(error_msg)
            return Command(
                goto="widget_supervisor",
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "task_status": "failed",
                    "updated_at": datetime.now(),
                },
            )