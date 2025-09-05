"""Database operations tool for top-level supervisor.

Following CLAUDE.md guidelines, this module contains ONLY agent tools that wrap
helper functions from src/actions/. All actual database logic is in src/actions/.
"""

from typing import Dict, Any, List
from datetime import datetime

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

from agent.models import TopLevelSupervisorState
from actions.dashboard import update_widgets_configuration_status, update_widgets_order_and_configuration
from actions.widget_ordering import analyze_and_order_dashboard_widgets

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
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
        completed_widget_ids = getattr(state, "completed_widget_ids", {}) or {}

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

        logger.info(
            f"Found {len(widget_ids_to_configure)} widget IDs to finalize: {widget_ids_to_configure}"
        )

        # Remove duplicates while preserving order (though there shouldn't be any)
        unique_widget_ids = list(dict.fromkeys(widget_ids_to_configure))

        # Step 1: Use LLM-based smart ordering to arrange widgets optimally
        # This will analyze existing widgets and determine the best order for new widgets
        try:
            logger.info(
                f"Starting LLM-based widget ordering analysis for dashboard {state.dashboard_id}"
            )

            ordering_result = analyze_and_order_dashboard_widgets(
                dashboard_id=state.dashboard_id,
                new_widget_ids=unique_widget_ids,
                chat_id=state.chat_id,
                user_id=state.user_id,
            )

            if ordering_result["success"]:
                logger.info(
                    f"✅ Smart widget ordering completed: {ordering_result['reasoning']}"
                )

                success_msg = f"✅ WIDGETS FINALIZED WITH SMART ORDERING: Successfully ordered and configured {ordering_result['updated_count']} widgets on the dashboard.\n\n"
                success_msg += (
                    f"📊 **Ordering Strategy**: {ordering_result['reasoning']}\n\n"
                )
                success_msg += (
                    f"Widgets finalized from completed_widget_ids:\n"
                    + "\n".join(task_summaries)
                )
                success_msg += f"\n\n🎯 All widgets are now live and optimally positioned for user experience."

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
                # Fallback to simple sequential ordering if smart ordering fails
                logger.warning(
                    f"Smart ordering failed: {ordering_result['message']}. Falling back to sequential ordering."
                )

                # Create simple sequential widget updates with order values
                fallback_widget_updates = []
                for i, widget_id in enumerate(unique_widget_ids, start=1):
                    fallback_widget_updates.append({
                        "id": widget_id,
                        "order": i,
                        "is_configured": True,
                    })

                config_result = update_widgets_order_and_configuration(fallback_widget_updates)

                if config_result["success"]:
                    logger.info(
                        f"Successfully configured {config_result['updated_count']} widgets as live (fallback)"
                    )

                    success_msg = f"✅ WIDGETS FINALIZED: Successfully configured {config_result['updated_count']} widgets as live on the dashboard.\n\n"
                    success_msg += f"⚠️ Note: Smart ordering was not available, widgets use sequential ordering (1, 2, 3...).\n\n"
                    success_msg += (
                        f"Widgets finalized from completed_widget_ids:\n"
                        + "\n".join(task_summaries)
                    )
                    success_msg += (
                        f"\n\n🎯 All widgets are now live and available to users."
                    )

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
                    error_msg = f"Both smart ordering and fallback configuration failed: {config_result.get('message', 'Unknown error')}"
                    logger.error(error_msg)

                    return Command(
                        update={
                            "error_messages": getattr(state, "error_messages", [])
                            + [error_msg],
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

        except Exception as ordering_error:
            error_msg = f"Error in smart widget ordering: {str(ordering_error)}"
            logger.error(error_msg)

            # Fallback to sequential ordering
            try:
                logger.info("Attempting fallback to sequential widget ordering...")
                
                # Create simple sequential widget updates with order values
                fallback_widget_updates = []
                for i, widget_id in enumerate(unique_widget_ids, start=1):
                    fallback_widget_updates.append({
                        "id": widget_id,
                        "order": i,
                        "is_configured": True,
                    })

                config_result = update_widgets_order_and_configuration(fallback_widget_updates)

                if config_result["success"]:
                    logger.info(
                        f"Fallback configuration successful: {config_result['updated_count']} widgets configured"
                    )

                    success_msg = f"✅ WIDGETS FINALIZED: Successfully configured {config_result['updated_count']} widgets as live on the dashboard.\n\n"
                    success_msg += f"⚠️ Note: Smart ordering failed ({str(ordering_error)}), using sequential ordering (1, 2, 3...).\n\n"
                    success_msg += (
                        f"Widgets finalized from completed_widget_ids:\n"
                        + "\n".join(task_summaries)
                    )
                    success_msg += (
                        f"\n\n🎯 All widgets are now live and available to users."
                    )

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
                    fallback_error = f"Fallback sequential ordering also failed: {config_result.get('message', 'Unknown error')}"
                    logger.error(fallback_error)

                    return Command(
                        update={
                            "error_messages": getattr(state, "error_messages", [])
                            + [error_msg, fallback_error],
                            "messages": [
                                ToolMessage(
                                    content=f"❌ WIDGET FINALIZATION ERROR: {error_msg}\n\nFallback also failed: {fallback_error}\n\nWidgets may have been created but could not be marked as live.",
                                    tool_call_id=tool_call_id,
                                )
                            ],
                            "supervisor_status": "failed",
                            "updated_at": datetime.now(),
                        }
                    )

            except Exception as fallback_error:
                final_error = (
                    f"Both smart ordering and fallback failed: {str(fallback_error)}"
                )
                logger.error(final_error)

                return Command(
                    update={
                        "error_messages": getattr(state, "error_messages", [])
                        + [error_msg, final_error],
                        "messages": [
                            ToolMessage(
                                content=f"❌ CRITICAL ERROR: {final_error}\n\nWidgets may have been created but could not be finalized.",
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
                "error_messages": getattr(state, "error_messages", []) + [error_msg],
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
