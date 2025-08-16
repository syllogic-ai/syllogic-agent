"""Top Level Supervisor Agent using create_react_agent pattern.

This supervisor orchestrates tasks across all specialized agent teams,
analyzes user requests, reads available data, and delegates appropriate tasks.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent, InjectedState
from typing_extensions import Annotated

from agent.models import TopLevelSupervisorState, WidgetAgentState
from .tools.data_reader import get_available_data
from .tools.task_manager import create_task, update_task_status, get_pending_tasks

logger = logging.getLogger(__name__)


@tool
def analyze_available_data(
    dashboard_id: str,
    state: Annotated[TopLevelSupervisorState, InjectedState]
) -> str:
    """Analyze what data is available for the given dashboard."""
    try:
        data_info = get_available_data(dashboard_id)
        
        # Update state with available data
        state.available_files = data_info["available_files"]
        state.available_data_summary = data_info["data_summary"]
        
        return f"Available data analysis completed:\n{data_info['data_summary']}"
        
    except Exception as e:
        logger.error(f"Error analyzing available data: {e}")
        return f"Error analyzing available data: {str(e)}"


@tool
def delegate_widget_task(
    task_instructions: str,
    operation: str,
    widget_type: str,
    title: str,
    description: str,
    file_ids: List[str],
    state: Annotated[TopLevelSupervisorState, InjectedState]
) -> str:
    """Delegate a widget-related task to the widget_agent_team.
    
    Args:
        task_instructions: Detailed instructions for the widget task
        operation: Widget operation (CREATE, UPDATE, DELETE)
        widget_type: Type of widget (line, bar, pie, area, radial, kpi, table)
        title: Widget title
        description: Widget description
        file_ids: List of file IDs to use for the widget
    """
    try:
        # Prepare WidgetAgentState initialization data
        widget_agent_state_data = {
            "task_instructions": task_instructions,
            "user_prompt": state.user_prompt,
            "operation": operation,
            "widget_type": widget_type,
            "title": title,
            "description": description,
            "dashboard_id": state.dashboard_id,
            "chat_id": state.chat_id,
            "file_ids": file_ids
        }
        
        # Create and add the task
        state = create_task(
            state=state,
            task_type="widget_operation",
            target_agent="widget_agent_team",
            task_instructions=task_instructions,
            widget_agent_state_data=widget_agent_state_data
        )
        
        logger.info(f"Delegated widget task: {operation} {widget_type} widget")
        return f"Successfully delegated {operation} task for {widget_type} widget '{title}' to widget_agent_team"
        
    except Exception as e:
        logger.error(f"Error delegating widget task: {e}")
        return f"Error delegating widget task: {str(e)}"


@tool
def check_task_status(
    state: Annotated[TopLevelSupervisorState, InjectedState]
) -> str:
    """Check the status of all delegated tasks."""
    try:
        if not state.delegated_tasks:
            return "No tasks have been delegated yet."
        
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
        if state.all_tasks_completed:
            status_report.append("✅ All tasks are completed!")
        
        return "\n".join(status_report)
        
    except Exception as e:
        logger.error(f"Error checking task status: {e}")
        return f"Error checking task status: {str(e)}"


@tool
def finalize_response(
    final_message: str,
    state: Annotated[TopLevelSupervisorState, InjectedState]
) -> str:
    """Finalize the response when all tasks are completed."""
    try:
        state.final_response = final_message
        state.supervisor_status = "completed"
        state.updated_at = datetime.now()
        
        logger.info("Top level supervisor completed all tasks")
        return f"Response finalized: {final_message}"
        
    except Exception as e:
        logger.error(f"Error finalizing response: {e}")
        return f"Error finalizing response: {str(e)}"


# Define the tools available to the supervisor
supervisor_tools = [
    analyze_available_data,
    delegate_widget_task,
    check_task_status,
    finalize_response
]

# Create the supervisor agent using create_react_agent
def create_top_level_supervisor(model_name: str = "anthropic:claude-3-5-sonnet-latest"):
    """Create the top-level supervisor agent."""
    
    system_prompt = """You are the Top Level Supervisor for a dashboard and widget management system.

Your responsibilities:
1. Analyze user requests and understand what they want to accomplish
2. Read and understand available data sources 
3. Break down complex requests into specific tasks
4. Delegate appropriate tasks to specialized agent teams
5. Monitor task progress and coordinate between teams
6. Provide final responses when all tasks are completed

Available Agent Teams:
- widget_agent_team: Handles widget creation, updates, and deletion operations

When you receive a user request:
1. FIRST: Use analyze_available_data to understand what data is available
2. THEN: Analyze the user prompt to determine what tasks need to be completed
3. DELEGATE: Use delegate_widget_task for any widget-related operations
4. MONITOR: Use check_task_status to track progress
5. FINALIZE: Use finalize_response when all tasks are completed

For widget operations, you must specify:
- operation: CREATE, UPDATE, or DELETE
- widget_type: line, bar, pie, area, radial, kpi, or table
- title: Clear widget title
- description: Widget description
- file_ids: List of relevant file IDs from available data

Always be thorough in your analysis and clear in your task delegation. 
Ensure all necessary information is provided to the specialized agents."""

    return create_react_agent(
        model=model_name,
        tools=supervisor_tools,
        prompt=system_prompt
    )


def top_level_supervisor(state: TopLevelSupervisorState) -> Dict[str, Any]:
    """Main entry point for the top level supervisor node.
    
    This function creates and invokes the supervisor agent, handling the
    orchestration of tasks across specialized agent teams.
    """
    try:
        # Create the supervisor agent
        supervisor_agent = create_top_level_supervisor()
        
        # Prepare the message for the agent
        user_message = HumanMessage(
            content=f"""
User Request: {state.user_prompt}
Dashboard ID: {state.dashboard_id}
Chat ID: {state.chat_id}
Request ID: {state.request_id}
User ID: {state.user_id}

Please analyze this request and coordinate the necessary tasks to fulfill it.
Start by analyzing the available data, then determine what needs to be done.
"""
        )
        
        # Invoke the supervisor agent
        result = supervisor_agent.invoke({
            "messages": [user_message]
        })
        
        # Extract the response
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_content = last_message.content
            else:
                response_content = str(last_message)
        else:
            response_content = "Task analysis completed"
        
        # Update state
        state.current_reasoning = response_content
        state.updated_at = datetime.now()
        
        return {
            "current_reasoning": response_content,
            "supervisor_status": state.supervisor_status,
            "updated_at": state.updated_at
        }
        
    except Exception as e:
        logger.error(f"Error in top level supervisor: {e}")
        state.error_messages.append(f"Supervisor error: {str(e)}")
        state.supervisor_status = "failed"
        state.updated_at = datetime.now()
        
        return {
            "error_messages": state.error_messages,
            "supervisor_status": "failed",
            "updated_at": state.updated_at
        }