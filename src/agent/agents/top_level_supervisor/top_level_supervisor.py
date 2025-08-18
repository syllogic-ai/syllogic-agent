"""Top Level Supervisor Agent using create_react_agent pattern.

This supervisor orchestrates tasks across all specialized agent teams,
analyzes user requests, reads available data, and delegates appropriate tasks.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

from agent.models import TopLevelSupervisorState, WidgetAgentState
from .tools.data_reader import get_available_data
from .tools.task_manager import create_task, update_task_status, get_pending_tasks
from .structured_output import SupervisorResponse, TaskCreationPlan, SupervisorDecision

logger = logging.getLogger(__name__)


@tool
def analyze_available_data(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Analyze what data is available for the dashboard. Takes no parameters - uses injected state."""
    try:
        # Get dashboard_id from state
        dashboard_id = state.dashboard_id
        
        data_info = get_available_data(dashboard_id)
        
        success_message = f"Available data analysis completed:\n{data_info['data_summary']}"
        
        # Return Command to update state properly
        return Command(
            update={
                "available_files": data_info["available_files"],
                "available_data_summary": data_info["data_summary"],
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
def delegate_widget_task(
    task_instructions: str,
    operation: str,
    widget_type: str,
    title: str,
    description: str,
    file_ids: List[str],
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    widget_id: Optional[str] = None
) -> Command:
    """Delegate a widget-related task to the widget_agent_team.
    
    Args:
        task_instructions: Detailed instructions for the widget task
        operation: Widget operation (CREATE, UPDATE, DELETE)
        widget_type: Type of widget (line, bar, pie, area, radial, kpi, table)
        title: Widget title
        description: Widget description
        file_ids: List of file IDs to use for the widget
        widget_id: Widget ID for UPDATE/DELETE operations or context reference
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
        
        # Add widget_id if provided
        if widget_id:
            widget_agent_state_data["widget_id"] = widget_id
        
        # Create and add the task (this modifies the state in place)
        updated_state = create_task(
            state=state,
            target_agent="widget_agent_team",
            task_instructions=task_instructions,
            widget_type=widget_type,
            operation=operation,
            file_ids=file_ids,
            widget_id=widget_id,
            widget_agent_state_data=widget_agent_state_data
        )
        
        success_message = f"Successfully delegated {operation} task for {widget_type} widget '{title}' to widget_agent_team"
        
        logger.info(f"Delegated widget task: {operation} {widget_type} widget")
        
        return Command(
            update={
                "delegated_tasks": updated_state.delegated_tasks,
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=success_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error delegating widget task: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


@tool
def check_task_status(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Check the status of all delegated tasks."""
    try:
        if not state.delegated_tasks:
            message = "No tasks have been delegated yet."
            return Command(
                update={
                    "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
                }
            )
        
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
        all_completed = len(completed_tasks) == len(state.delegated_tasks) and len(state.delegated_tasks) > 0
        if all_completed:
            status_report.append("✅ All tasks are completed!")
        
        status_message = "\n".join(status_report)
        
        return Command(
            update={
                "all_tasks_completed": all_completed,
                "messages": [ToolMessage(content=status_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error checking task status: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


@tool
def execute_widget_tasks(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Execute pending widget tasks by delegating to widget_agent_team."""
    try:
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
        
        # Mark the first task as in_progress
        current_task = pending_tasks[0]
        current_task.task_status = "in_progress"
        current_task.started_at = datetime.now()
        
        execution_message = f"DELEGATE_TO_WIDGET_TEAM: {current_task.task_id}"
        reasoning = f"Executing widget task: {current_task.task_instructions}"
        
        logger.info(f"Executing task {current_task.task_id} with widget_agent_team")
        
        # This will signal to the graph to route to widget_agent_team
        return Command(
            update={
                "current_reasoning": reasoning,
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
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


@tool
def finalize_response(
    final_message: str,
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Finalize the response when all tasks are completed."""
    try:
        completion_message = f"Response finalized: {final_message}"
        
        logger.info("Top level supervisor completed all tasks")
        
        return Command(
            update={
                "final_response": final_message,
                "supervisor_status": "completed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=completion_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error finalizing response: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


# Define the tools available to the supervisor
supervisor_tools = [
    analyze_available_data,
    delegate_widget_task,
    execute_widget_tasks,
    check_task_status,
    finalize_response
]

# Create the supervisor agent using create_react_agent with structured output
def create_top_level_supervisor(model_name: str = "openai:gpt-4o-mini"):
    """Create the top-level supervisor agent with structured output."""
    
    system_prompt = """You are the Top Level Supervisor for a dashboard and widget management system.

Your responsibilities:
1. Analyze user requests and understand what they want to accomplish
2. Read and understand available data sources 
3. Strategically plan widget creation to comprehensively answer the user's needs
4. Delegate complementary tasks that work together to fulfill the request
5. Monitor task progress and coordinate between teams
6. Provide structured responses about your progress and decisions

Available Agent Teams:
- widget_agent_team: Handles widget creation, updates, and deletion operations

CRITICAL TASK PLANNING RULES:
1. NEVER create duplicate or redundant tasks
2. Tasks must complement each other to provide a complete answer
3. For specific requests (e.g., "create a bar chart"), create exactly what was asked
4. For vague requests, think strategically about what widgets would best answer the user's question

WIDGET STRATEGY GUIDELINES:
- Single chart request: Create exactly one widget as requested
- Exploratory questions: Consider multiple complementary widgets (e.g., overview + drill-down)
- Dashboard requests: Plan 3-5 widgets that tell a complete story
- Update/delete requests: Use context_widget_ids to identify target widgets

STRUCTURED OUTPUT REQUIREMENT:
You MUST always provide responses in the specified structured format. Your response should include:
- Current status (analyzing, planning, delegating, monitoring, completed, failed)
- Clear human-readable message about what you're doing
- Specific decision details when planning
- Task creation plans when delegating
- Summary of progress when monitoring

When you receive a user request:
1. FIRST: Use analyze_available_data to understand what data is available
2. THEN: Analyze the user prompt and plan your widget strategy:
   - What is the user really trying to understand?
   - What widgets would best answer their question?
   - How can widgets complement each other?
3. CHECK: Review any existing delegated tasks to avoid duplicates
4. DELEGATE: Use delegate_widget_task for each unique widget needed
5. MONITOR: Use check_task_status to track progress
6. RESPOND: Always provide structured output about your decisions and progress

For each widget operation, you must specify:
- operation: CREATE, UPDATE, or DELETE
- widget_type: line, bar, pie, area, radial, kpi, or table (choose based on data and purpose)
- title: Clear, specific widget title that indicates its purpose
- description: Widget description explaining what insights it provides
- file_ids: List of relevant file IDs from available data
- widget_id: Required for UPDATE/DELETE operations (use context_widget_ids)

SMART DELEGATION EXAMPLES:
- "Show me sales trends" → 1 line chart of sales over time
- "Analyze our performance" → Multiple widgets: KPI summary, trends chart, breakdown by category
- "Update the revenue chart" → 1 UPDATE operation using provided widget_id
- "What's our best selling product?" → Bar chart + KPI widget for comprehensive answer

Always think before delegating: Does this task complement my existing tasks? Am I avoiding duplicates? Will these widgets together answer the user's question completely?

IMPORTANT: Always respond with the structured SupervisorResponse format providing clear status, message, and relevant details."""

    return create_react_agent(
        model=model_name,
        tools=supervisor_tools,
        prompt=system_prompt
        # Temporarily removing response_format to avoid validation errors
        # response_format=SupervisorResponse
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
        
        # Invoke the supervisor agent with complete state
        agent_input = {
            "messages": [user_message],
            # Pass through all the required state fields for tool injection
            "user_prompt": state.user_prompt,
            "user_id": state.user_id,
            "dashboard_id": state.dashboard_id, 
            "chat_id": state.chat_id,
            "request_id": state.request_id,
            "file_ids": state.file_ids,
            "context_widget_ids": state.context_widget_ids,
            "available_files": state.available_files,
            "available_data_summary": state.available_data_summary,
            "delegated_tasks": state.delegated_tasks,
            "supervisor_status": state.supervisor_status,
            "current_reasoning": state.current_reasoning,
            "final_response": state.final_response,
            "error_messages": state.error_messages,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "all_tasks_completed": state.all_tasks_completed
        }
        
        result = supervisor_agent.invoke(agent_input)
        
        # Extract structured response
        structured_response = None
        response_content = "Task analysis completed"
        
        if result and "structured_response" in result:
            structured_response = result["structured_response"]
            response_content = structured_response.message
            
            # Update supervisor status based on structured response
            if structured_response.status in ["analyzing", "planning", "delegating", "monitoring", "completed", "failed"]:
                state.supervisor_status = structured_response.status
                
        elif result and "messages" in result:
            # Fallback to message content if no structured response
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_content = last_message.content
            else:
                response_content = str(last_message)
        
        # Update state
        state.current_reasoning = response_content
        state.updated_at = datetime.now()
        
        # Return both regular state updates and structured response for external use
        return {
            "current_reasoning": response_content,
            "supervisor_status": state.supervisor_status,
            "updated_at": state.updated_at,
            "structured_response": structured_response
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