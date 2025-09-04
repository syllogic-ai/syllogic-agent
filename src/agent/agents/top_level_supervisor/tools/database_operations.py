"""Database operations tool for top-level supervisor.

Following CLAUDE.md guidelines, this module contains ONLY agent tools that wrap
helper functions from src/actions/. All actual database logic is in src/actions/.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

from agent.models import TopLevelSupervisorState
from actions.dashboard import update_widgets_configuration_status

logger = logging.getLogger(__name__)


@tool  
def finalize_created_widgets(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Unified tool that finds widgets created by completed widget agent tasks and marks them as configured (live).
    Uses the completed_widget_ids mapping from state to find widget IDs that need to be marked as live.
    
    This is an agent tool wrapper around helper functions in src/actions/dashboard.
    """
    try:
        # Get completed widget IDs from state mapping
        completed_widget_ids = getattr(state, 'completed_widget_ids', {}) or {}
        
        if not completed_widget_ids:
            logger.info("No completed widget IDs found in state")
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="No completed widget IDs found to finalize.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "updated_at": datetime.now(),
                }
            )
        
        # Extract widget IDs and create task summaries
        widget_ids_to_configure = list(completed_widget_ids.values())
        task_summaries = []
        
        for task_id, widget_id in completed_widget_ids.items():
            task_summaries.append(f"- Task {task_id}: Widget {widget_id}")
        
        logger.info(f"Found {len(widget_ids_to_configure)} widget IDs to finalize: {widget_ids_to_configure}")
        
        # Remove duplicates while preserving order (though there shouldn't be any)
        unique_widget_ids = list(dict.fromkeys(widget_ids_to_configure))
        
        # Update is_configured=true for all widgets created by completed tasks
        # Using helper function from actions.dashboard (follows CLAUDE.md guidelines)
        try:
            config_result = update_widgets_configuration_status(unique_widget_ids, is_configured=True)
            
            if config_result["success"]:
                logger.info(f"Successfully configured {config_result['updated_count']} widgets as live")
                
                success_msg = f"✅ WIDGETS FINALIZED: Successfully configured {config_result['updated_count']} widgets as live on the dashboard.\n\n"
                success_msg += f"Widgets finalized from completed_widget_ids:\n" + "\n".join(task_summaries)
                success_msg += f"\n\n🎯 All widgets are now live and available to users."
                
                # Clear the completed_widget_ids after successful finalization
                return Command(
                    update={
                        "completed_widget_ids": {},  # Clear the mapping after finalization
                        "messages": [
                            ToolMessage(
                                content=success_msg,
                                tool_call_id=tool_call_id,
                            )
                        ],
                        "supervisor_status": "completed",
                        "all_tasks_completed": True,
                        "updated_at": datetime.now(),
                    }
                )
            else:
                error_msg = f"Failed to configure widgets: {config_result.get('message', 'Unknown error')}"
                logger.error(error_msg)
                
                return Command(
                    update={
                        "error_messages": getattr(state, 'error_messages', []) + [error_msg],
                        "messages": [
                            ToolMessage(
                                content=f"⚠️ WIDGET FINALIZATION FAILED: {error_msg}\n\nWidgets were created but are not yet live.",
                                tool_call_id=tool_call_id,
                            )
                        ],
                        "supervisor_status": "failed",
                        "updated_at": datetime.now(),
                    }
                )
                
        except Exception as config_error:
            error_msg = f"Error updating widget configuration status: {str(config_error)}"
            logger.error(error_msg)
            
            return Command(
                update={
                    "error_messages": getattr(state, 'error_messages', []) + [error_msg],
                    "messages": [
                        ToolMessage(
                            content=f"❌ WIDGET FINALIZATION ERROR: {error_msg}\n\nWidgets may have been created but could not be marked as live.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "supervisor_status": "failed",
                    "updated_at": datetime.now(),
                }
            )
            
    except Exception as e:
        error_msg = f"Failed to finalize created widgets: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": getattr(state, 'error_messages', []) + [error_msg],
                "messages": [
                    ToolMessage(
                        content=f"❌ WIDGET FINALIZATION FAILED: {error_msg}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
            }
        )