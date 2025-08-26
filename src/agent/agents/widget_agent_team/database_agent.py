"""Database operations agent for widget CRUD operations."""

import os
import sys
import uuid
from datetime import datetime

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent.models import CreateWidgetInput, UpdateWidgetInput, WidgetAgentState

from actions.utils import import_actions_dashboard
from .tools.widget_summary import generate_widget_summary


class DatabaseAgent:
    """Database operations agent that handles CREATE/UPDATE/DELETE operations for widgets."""

    def __init__(self):
        """Initialize database agent."""
        pass

    def perform_db_operations(self, state: WidgetAgentState) -> Command:
        """
        Database operations that handles CREATE/UPDATE/DELETE operations for widgets.
        Uses create_widget, update_widget, delete_widget from dashboard.py.
        """
        try:
            # Import dashboard functions using robust import
            try:
                dashboard_module = import_actions_dashboard()
                create_widget = dashboard_module.create_widget
                update_widget = dashboard_module.update_widget
                delete_widget = dashboard_module.delete_widget
            except ImportError as e:
                error_msg = f"Failed to import actions.dashboard module: {str(e)}"
                return Command(
                    goto="widget_supervisor",
                    update={
                        "error_messages": state.error_messages + [error_msg],
                        "task_status": "failed",
                        "updated_at": datetime.now(),
                    },
                )

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
                
                # Create widget input using unified widget_config
                create_input = CreateWidgetInput(
                    dashboard_id=dashboard_id,
                    title=title,
                    widget_type=widget_type,
                    config=state.widget_config or {},  # Use unified widget_config
                    widget_id=widget_id,  # Pass the widget_id to be used
                    chat_id=state.chat_id,
                    description=state.description,
                    summary=widget_summary,  # Add the generated summary
                    data={"data": state.widget_config.get("data", []) if state.widget_config else []},  # Extract data from widget_config
                )
                
                # Create widget in database
                created_widget = create_widget(create_input)
                
                success_msg = f"✅ DATABASE OPERATION COMPLETE: Successfully created widget '{title}' with ID: {created_widget.id}. The entire widget creation task is now COMPLETED. All data has been processed, validated, and persisted to the database."
                return Command(
                    goto="widget_supervisor",
                    update={
                        "widget_id": created_widget.id,
                        "updated_at": datetime.now(),
                        "widget_creation_completed": True,  # Clear completion signal
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
                
                # Update existing widget using unified widget_config
                update_input = UpdateWidgetInput(
                    widget_id=state.widget_id,
                    title=title,
                    widget_type=widget_type,
                    config=state.widget_config or {},  # Use unified widget_config
                    description=state.description,
                    summary=widget_summary,  # Add the updated summary
                    data={"data": state.widget_config.get("data", []) if state.widget_config else []},  # Extract data from widget_config
                )
                
                # Update widget in database
                updated_widget = update_widget(update_input)
                
                success_msg = f"✅ DATABASE OPERATION COMPLETE: Successfully updated widget '{title}' with ID: {state.widget_id}. The entire widget update task is now COMPLETED. All data has been processed, validated, and persisted to the database."
                return Command(
                    goto="widget_supervisor",
                    update={
                        "updated_at": datetime.now(),
                        "widget_update_completed": True,  # Clear completion signal
                        "messages": [
                            ToolMessage(
                                content=success_msg,
                                tool_call_id="db_update_complete",
                            )
                        ],
                    },
                )
                
            elif operation == "DELETE":
                # Delete widget from database
                deletion_success = delete_widget(state.widget_id)
                
                if deletion_success:
                    success_msg = f"✅ DATABASE OPERATION COMPLETE: Successfully deleted widget with ID: {state.widget_id}. The entire widget deletion task is now COMPLETED."
                    return Command(
                        goto="widget_supervisor",
                        update={
                            "updated_at": datetime.now(),
                            "widget_deletion_completed": True,  # Clear completion signal
                            "messages": [
                                ToolMessage(
                                    content=success_msg,
                                    tool_call_id="db_delete_complete",
                                )
                            ],
                        },
                    )
                else:
                    error_msg = f"Failed to delete widget with ID: {state.widget_id} (widget not found)"
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
            error_msg = f"Database operations error: {str(e)}"
            return Command(
                goto="widget_supervisor",
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "task_status": "failed",
                    "updated_at": datetime.now(),
                },
            )