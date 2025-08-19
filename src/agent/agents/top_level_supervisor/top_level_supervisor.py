"""Top Level Supervisor Agent using create_react_agent pattern.

This supervisor orchestrates tasks across all specialized agent teams,
analyzes user requests, reads available data, and delegates appropriate tasks.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command, Send
from langgraph.graph import END
from typing_extensions import Annotated

from agent.models import TopLevelSupervisorState, WidgetAgentState, DelegatedTask
from .tools.data_reader import get_available_data
from .tools.task_manager import update_task_status, get_pending_tasks
from .structured_output import SupervisorResponse, TaskCreationPlan, SupervisorDecision, TaskCreationRequest

logger = logging.getLogger(__name__)


@tool
def analyze_available_data(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Analyze what data is available for the dashboard. Takes no parameters - uses injected state."""
    try:
        # Check if data analysis has already been completed
        if state.available_data_summary and state.available_files:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content="Data analysis already completed. Available data summary is ready.",
                        tool_call_id=tool_call_id
                    )],
                }
            )
        
        # Get dashboard_id from state
        dashboard_id = state.dashboard_id
        
        data_info = get_available_data(dashboard_id)
        
        success_message = f"""✅ **DATA ANALYSIS COMPLETE**

{data_info['data_summary']}

**📋 NEXT STEP:** Use the plan_widget_tasks tool to intelligently analyze the user request and create the appropriate DelegatedTask objects for the widget_agent_team."""
        
        # Return Command to update state properly
        return Command(
            update={
                "available_files": data_info["available_files"],
                "file_schemas": data_info["file_schemas"],  # Store detailed schema info
                "available_data_summary": data_info["data_summary"],
                "supervisor_status": "delegating",  # Move to next phase
                "updated_at": datetime.now(),
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
def plan_widget_tasks(
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """AI-powered tool that analyzes user query and available data to create multiple DelegatedTask objects."""
    try:
        # Check if planning has already been done
        if state.delegated_tasks:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content=f"Task planning already completed. {len(state.delegated_tasks)} task(s) have been planned and are ready for execution.",
                        tool_call_id=tool_call_id
                    )],
                }
            )
        
        # Use AI with structured output to create intelligent task plan
        planning_llm = ChatOpenAI(model="gpt-4o-mini")
        planning_llm_structured = planning_llm.with_structured_output(TaskCreationPlan)
        
        # Create focused planning prompt with emphasis on minimal task creation
        planning_prompt = f"""
You are an expert data visualization and dashboard planning AI. Analyze the user's request and available data to create a MINIMAL, targeted plan for widget creation.

**USER REQUEST:**
{state.user_prompt}

**AVAILABLE DATA:**
Data Summary: {state.available_data_summary or "No data summary available"}
Available Files: {len(state.available_files)} file(s) 
File IDs: {state.available_files}
Dashboard ID: {state.dashboard_id}
Chat ID: {state.chat_id}

**CRITICAL TASK CREATION RULES:**
🎯 **ONLY CREATE TASKS THAT DIRECTLY ANSWER THE USER QUERY**
🎯 **NO NEED TO CREATE MULTIPLE OR TOO MANY WIDGETS IF THE USER QUERY CAN BE ANSWERED BY JUST ONE OR A COUPLE OF TASKS**
🎯 **QUALITY OVER QUANTITY - PREFER FEWER, MORE FOCUSED WIDGETS**

**MINIMAL PLANNING STRATEGY:**
1. Identify the CORE question the user is asking
2. Determine the MINIMUM number of widgets needed to answer that question
3. Create ONLY the essential tasks - avoid "nice to have" widgets
4. Each task must directly contribute to answering the user's specific request

**WIDGET TYPE GUIDANCE:**
- bar: Comparisons between categories, rankings, grouped data
- line: Trends over time, time series analysis, progression  
- pie: Parts of a whole, percentage breakdowns, composition
- area: Cumulative values over time, stacked trends, volume
- radial: Progress indicators, completion rates, gauges
- kpi: Single important metrics, key performance numbers, summary stats
- table: Detailed data exploration, multi-dimensional data, raw data display

**FOCUSED TASK CREATION EXAMPLES:**
- "Show me sales trends" → 1 line chart (STOP - that's sufficient)
- "What's our revenue?" → 1 KPI widget (STOP - that answers it)
- "Create a bar chart of products" → 1 bar chart (STOP - exactly what was requested)
- "Compare this quarter vs last" → 1-2 widgets maximum (comparison chart + maybe summary KPI)

**AVOID OVER-PLANNING:**
❌ Don't create comprehensive dashboards unless explicitly requested
❌ Don't add "supporting" widgets unless truly necessary
❌ Don't create multiple views of the same data
❌ Don't anticipate unstated user needs

You must provide a structured TaskCreationPlan with:
1. tasks: List of TaskCreationRequest objects (KEEP THIS MINIMAL!)
2. strategy_summary: Explanation of why this minimal set answers the user's question
3. duplicate_check: Confirmation no duplicates are being created

For each task, ensure:
- widget_type: Choose the most appropriate type based on data and purpose
- operation: Always "CREATE" for new widgets  
- title: Clear, specific title indicating the widget's purpose
- description: What insights this widget provides to answer the user's question
- file_ids: Use the available file IDs provided in the context
- task_instructions: Detailed instructions for widget_agent_team execution
"""

        # Get structured AI planning response
        task_plan = planning_llm_structured.invoke(planning_prompt)
        
        # Create DelegatedTask objects from the AI plan
        created_task_names = []
        updated_tasks = state.delegated_tasks.copy()  # Start with existing tasks
        
        for task_request in task_plan.tasks:
            # Create each task directly using DelegatedTask model
            new_task = DelegatedTask(
                target_agent="widget_agent_team",
                task_instructions=task_request.task_instructions,
                widget_type=task_request.widget_type,
                operation=task_request.operation,
                file_ids=task_request.file_ids,
                title=task_request.title,
                description=task_request.description,
                user_prompt=state.user_prompt,
                dashboard_id=state.dashboard_id,
                chat_id=state.chat_id,
                widget_id=task_request.widget_id,
                task_status="pending"  # Always create as pending
            )
            updated_tasks.append(new_task)
            created_task_names.append(f"{task_request.title} ({task_request.widget_type})")
        
        # Create comprehensive success message
        success_message = f"""✅ **AI PLANNING COMPLETE**

**Strategy:** {task_plan.strategy_summary}

**Tasks Created:** {len(task_plan.tasks)} 
{chr(10).join([f"• {name}" for name in created_task_names])}

**Duplicate Check:** {task_plan.duplicate_check}

**📋 NEXT STEP:** Use execute_widget_tasks to process these AI-planned tasks with the widget_agent_team."""

        return Command(
            update={
                "delegated_tasks": updated_tasks,  # CRITICAL: Include updated tasks in state
                "current_reasoning": f"AI planning completed: {len(task_plan.tasks)} widget tasks created based on intelligent analysis of user request and available data.",
                "supervisor_status": "monitoring",  # Ready to monitor task execution
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=success_message, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        error_msg = f"Error in AI planning: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


# delegate_widget_task REMOVED - redundant with plan_widget_tasks + execute_widget_tasks workflow


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
    """Execute pending widget tasks by directly invoking widget_agent_team subgraph."""
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
        
        # Process the first pending task (one at a time for now)
        current_task = pending_tasks[0]
        
        # Import widget graph here to avoid circular imports
        from agent.graph import build_widget_agent_graph
        
        # Create widget agent input from DelegatedTask data
        widget_input = {
            "messages": [],  # Required for LangGraph
            "remaining_steps": 10,  # Required for LangGraph
            "task_id": current_task.task_id,
            "task_instructions": current_task.task_instructions,
            "user_prompt": current_task.user_prompt,
            "operation": current_task.operation,
            "widget_type": current_task.widget_type,
            "dashboard_id": current_task.dashboard_id,
            "chat_id": current_task.chat_id,
            "file_ids": current_task.file_ids,
            "widget_id": current_task.widget_id or str(uuid.uuid4()),  # Generate ID if None
            "title": current_task.title,
            "description": current_task.description,
            "task_status": "in_progress",
            "created_at": current_task.created_at,
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
        
        logger.info(f"Invoking widget_agent_team for task {current_task.task_id}: {current_task.operation} {current_task.widget_type} widget")
        
        # Build and invoke widget agent subgraph
        widget_graph = build_widget_agent_graph()
        widget_result = widget_graph.invoke(widget_input)
        
        # Update task status based on widget result
        updated_tasks = []
        for task in state.delegated_tasks:
            if task.task_id == current_task.task_id:
                # Update task with result
                task.task_status = widget_result.get("task_status", "completed")
                task.completed_at = datetime.now()
                task.started_at = datetime.now()
                
                if widget_result.get("error_messages"):
                    task.error_message = "; ".join(widget_result["error_messages"])
                    task.task_status = "failed"
                else:
                    task.result = f"Widget {current_task.operation} operation completed successfully"
                    
            updated_tasks.append(task)
        
        execution_message = f"✅ **WIDGET TASK EXECUTED**\n\nCompleted {current_task.operation} {current_task.widget_type} widget task\nStatus: {widget_result.get('task_status', 'completed')}"
        
        logger.info(f"Widget task {current_task.task_id} completed with status: {widget_result.get('task_status')}")
        
        return Command(
            update={
                "delegated_tasks": updated_tasks,
                "supervisor_status": "analyzing",  # Return to analyzing to check for more tasks
                "current_reasoning": f"Widget task completed: {widget_result.get('task_status', 'completed')}",
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
                "supervisor_status": "failed",
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )


@tool
def generate_error_response(
    failed_tool: str,
    error_messages: List[str],
    user_prompt: str,
    state: Annotated[TopLevelSupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Generate an AI-powered error response when tools fail repeatedly."""
    try:
        # Create a user-friendly error message
        error_summary = f"I encountered repeated issues while trying to {failed_tool}. "
        
        if failed_tool == "analyze_available_data":
            error_summary += "I was unable to access or analyze the data files needed for your request. "
        elif failed_tool == "plan_widget_tasks":
            error_summary += "I had trouble planning the appropriate widgets for your request. "
        elif failed_tool == "delegate_widget_task":
            error_summary += "I couldn't successfully delegate the widget creation task. "
        else:
            error_summary += "There was a technical issue with the system. "
        
        # Add specific error details if available
        if error_messages:
            recent_errors = error_messages[-3:]  # Show last 3 errors
            error_details = "\\n".join([f"• {error}" for error in recent_errors])
            error_summary += f"\\n\\nTechnical details:\\n{error_details}"
        
        error_summary += "\\n\\nPlease try your request again later, or contact support if the issue persists."
        
        return Command(
            goto=END,
            update={
                "final_response": error_summary,
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=error_summary, tool_call_id=tool_call_id)],
            }
        )
        
    except Exception as e:
        fallback_message = f"I'm sorry, but I encountered technical difficulties while processing your request: '{user_prompt}'. Please try again later or contact support."
        logger.error(f"Error generating error response: {e}")
        return Command(
            goto=END,
            update={
                "final_response": fallback_message,
                "supervisor_status": "failed",
                "updated_at": datetime.now(),
                "messages": [ToolMessage(content=fallback_message, tool_call_id=tool_call_id)],
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
            goto=END,
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
    plan_widget_tasks,
    execute_widget_tasks,
    check_task_status,
    generate_error_response,
    finalize_response
]

# Create the supervisor agent using create_react_agent with structured output
def create_top_level_supervisor(model_name: str = "gpt-4o-mini"):
    """Create the top-level supervisor agent with structured output."""
    
    # Initialize OpenAI model following widget_agent_team pattern
    model = ChatOpenAI(model=model_name)
    
    system_prompt = """You are the Top Level Supervisor for a dashboard and widget management system.

Your responsibilities:
1. Analyze user requests and understand what they want to accomplish
2. Read and understand available data sources 
3. Create MINIMAL, focused widget plans that directly answer the user's question
4. Delegate only essential tasks - avoid over-planning or unnecessary widgets
5. Monitor task progress and coordinate between teams
6. Provide structured responses about your progress and decisions

Available Agent Teams:
- widget_agent_team: Handles widget creation, updates, and deletion operations

🎯 **CRITICAL TASK CREATION PHILOSOPHY:**
**ONLY CREATE TASKS THAT DIRECTLY ANSWER THE USER QUERY**
**NO NEED TO CREATE MULTIPLE OR TOO MANY WIDGETS IF THE USER QUERY CAN BE ANSWERED BY JUST ONE OR A COUPLE OF TASKS**

MINIMAL TASK PLANNING RULES:
1. NEVER create duplicate or redundant tasks
2. Create the MINIMUM number of widgets needed to answer the user's question
3. For specific requests (e.g., "create a bar chart"), create exactly one widget as requested
4. For analytical questions, create only essential widgets - usually 1-2 maximum
5. Avoid "comprehensive" or "supporting" widgets unless explicitly requested

FOCUSED WIDGET STRATEGY:
- Single chart request: Create exactly one widget (STOP - that's sufficient)
- Data question: Create one primary visualization that answers it directly
- Comparison request: Create one comparison widget (maybe + one summary KPI if essential)
- Update/delete requests: Use context_widget_ids to identify target widgets
- Avoid dashboard-style multi-widget responses unless user explicitly asks for a "dashboard"

STRUCTURED OUTPUT REQUIREMENT:
You MUST always provide responses in the specified structured format. Your response should include:
- Current status (analyzing, planning, delegating, monitoring, completed, failed)
- Clear human-readable message about what you're doing
- Specific decision details when planning
- Task creation plans when delegating
- Summary of progress when monitoring

WORKFLOW LOGIC - Follow this step by step:

1. **DATA ANALYSIS PHASE** (status: "analyzing"):
   - IF available_data_summary is empty or None, use analyze_available_data tool
   - IF available_data_summary exists, proceed to step 2 (data analysis complete)

2. **AI PLANNING PHASE** (status: "delegating"):
   - IF no delegated_tasks exist yet, use plan_widget_tasks tool
   - This AI-powered tool will analyze the user request and available data
   - It creates multiple DelegatedTask objects as needed to fully answer the user's question
   - The AI planning tool determines: widget types, titles, descriptions, and task instructions
   - IF delegated_tasks already exist, skip to step 3 (planning already complete)

3. **MONITORING PHASE** (status: "monitoring"):
   - Use check_task_status to track task progress
   - Use execute_widget_tasks to process pending tasks
   
4. **COMPLETION PHASE** (status: "completed"):
   - Use finalize_response when all tasks are completed

5. **ERROR HANDLING** (any status):
   - Check tool_failure_counts and last_failed_tool to identify repeated failures
   - If a tool has failed max_tool_retries times, use generate_error_response
   - Otherwise, you may retry the failed tool if it makes sense
   - Always consider whether retrying will help or if the issue is fundamental

CRITICAL: You are an AI agent - use your intelligence to analyze the user's request and available data, then make smart decisions about which widgets to create. Check your current supervisor_status, available_data_summary, delegated_tasks, and tool_failure_counts before deciding what to do next!

For each widget operation, you must specify:
- operation: CREATE, UPDATE, or DELETE
- widget_type: line, bar, pie, area, radial, kpi, or table (choose based on data and purpose)
- title: Clear, specific widget title that indicates its purpose
- description: Widget description explaining what insights it provides
- file_ids: List of relevant file IDs from available data
- widget_id: Required for UPDATE/DELETE operations (use context_widget_ids)

MINIMAL DELEGATION EXAMPLES:
- "Show me sales trends" → 1 line chart of sales over time (STOP - that answers it)
- "Analyze our performance" → 1 primary performance chart (avoid multiple widgets unless essential)
- "Update the revenue chart" → 1 UPDATE operation using provided widget_id
- "What's our best selling product?" → 1 bar chart showing top products (avoid additional KPIs unless requested)
- "Create a dashboard" → Multiple widgets allowed ONLY when user explicitly asks for "dashboard"

Always think before delegating: Is this the MINIMUM needed to answer the user's question? Am I avoiding over-planning? Will this single widget (or minimal set) fully satisfy their request?

IMPORTANT: Always respond with the structured SupervisorResponse format providing clear status, message, and relevant details."""

    return create_react_agent(
        model=model,  # Use the properly initialized ChatOpenAI model
        tools=supervisor_tools,
        prompt=system_prompt,
        state_schema=TopLevelSupervisorState  # Specify the state schema for InjectedState
        # Temporarily removing response_format to avoid validation errors
        # response_format=SupervisorResponse
    )


def handle_tool_failure(supervisor_state: TopLevelSupervisorState, error_message: str) -> Dict[str, Any]:
    """Handle tool failures with retry logic and error response generation."""
    try:
        # Extract tool name from error message if possible
        failed_tool = "unknown_tool"
        if "analyze_available_data" in error_message:
            failed_tool = "analyze_available_data"
        elif "plan_widget_tasks" in error_message:
            failed_tool = "plan_widget_tasks"
        elif "delegate_widget_task" in error_message:
            failed_tool = "delegate_widget_task"
        
        # Update failure counts
        current_failures = supervisor_state.tool_failure_counts.get(failed_tool, 0) + 1
        supervisor_state.tool_failure_counts[failed_tool] = current_failures
        supervisor_state.last_failed_tool = failed_tool
        
        # Add error to messages
        supervisor_state.error_messages.append(f"Tool {failed_tool} failed (attempt {current_failures}): {error_message}")
        
        # Check if we've exceeded retry limit
        if current_failures > supervisor_state.max_tool_retries:
            # Generate AI-powered error response
            error_summary = f"I encountered repeated issues with {failed_tool} after {current_failures} attempts. "
            
            if failed_tool == "analyze_available_data":
                error_summary += "I was unable to access or analyze the data files needed for your request. This might be due to database connectivity issues or missing data files."
            elif failed_tool == "plan_widget_tasks":
                error_summary += "I had trouble planning the appropriate widgets for your request. This might be due to complex requirements or system limitations."
            elif failed_tool == "delegate_widget_task":
                error_summary += "I couldn't successfully delegate the widget creation task. This might be due to resource constraints or system issues."
            else:
                error_summary += "There was a persistent technical issue with the system."
            
            error_summary += f"\\n\\nYour request: '{supervisor_state.user_prompt}'\\n\\nPlease try again later or contact support if the issue persists."
            
            return {
                "final_response": error_summary,
                "supervisor_status": "failed",
                "error_messages": supervisor_state.error_messages,
                "tool_failure_counts": supervisor_state.tool_failure_counts,
                "updated_at": datetime.now()
            }
        else:
            # Tool can still be retried
            retry_message = f"Tool {failed_tool} failed (attempt {current_failures}/{supervisor_state.max_tool_retries}). Will attempt retry if appropriate."
            
            return {
                "current_reasoning": retry_message,
                "supervisor_status": supervisor_state.supervisor_status,  # Keep current status for retry
                "error_messages": supervisor_state.error_messages,
                "tool_failure_counts": supervisor_state.tool_failure_counts,
                "last_failed_tool": failed_tool,
                "updated_at": datetime.now()
            }
            
    except Exception as e:
        logger.error(f"Error in handle_tool_failure: {e}")
        return {
            "final_response": f"I encountered technical difficulties while processing your request. Please try again later.",
            "supervisor_status": "failed",
            "error_messages": supervisor_state.error_messages + [f"Error handler failed: {str(e)}"],
            "updated_at": datetime.now()
        }


def top_level_supervisor(state) -> Dict[str, Any]:
    """Main entry point for the top level supervisor node.
    
    This function creates and invokes the supervisor agent, handling the
    orchestration of tasks across specialized agent teams.
    """
    supervisor_state = None  # Initialize to avoid UnboundLocalError
    try:
        # Convert dict state to TopLevelSupervisorState for proper tool injection
        if isinstance(state, dict):
            # Convert 'pending' status to 'analyzing' if present
            if state.get('supervisor_status') == 'pending':
                state['supervisor_status'] = 'analyzing'
            supervisor_state = TopLevelSupervisorState(**state)
        else:
            supervisor_state = state
            
        # Create the supervisor agent
        supervisor_agent = create_top_level_supervisor()
        
        # Prepare the message for the agent
        user_message = HumanMessage(
            content=f"""
User Request: {supervisor_state.user_prompt}
Dashboard ID: {supervisor_state.dashboard_id}
Chat ID: {supervisor_state.chat_id}
Request ID: {supervisor_state.request_id}
User ID: {supervisor_state.user_id}

Please analyze this request and coordinate the necessary tasks to fulfill it.
Start by analyzing the available data, then determine what needs to be done.
"""
        )
        
        # For InjectedState to work properly with create_react_agent, 
        # we need to pass the full state as the primary input
        agent_input = {
            **supervisor_state.model_dump(),  # Include all state fields
            "messages": [user_message],  # Add the user message to messages
        }
        
        # Invoke the agent with the complete state and handle potential failures
        try:
            result = supervisor_agent.invoke(agent_input)
        except Exception as tool_error:
            # Handle tool execution errors
            return handle_tool_failure(supervisor_state, str(tool_error))
        
        # Check if any tool in the result requested termination via Command(goto=END)
        # This happens when finalize_response or generate_error_response tools are called
        if result and "messages" in result:
            for message in result["messages"]:
                # Check if this is a ToolMessage from finalize_response or generate_error_response
                if hasattr(message, 'tool_call_id') and hasattr(message, 'content'):
                    if ("Response finalized:" in str(message.content) or 
                        "I encountered repeated issues" in str(message.content) or
                        "I'm sorry, but I encountered technical difficulties" in str(message.content)):
                        # These tools used goto=END, so we should terminate
                        logger.info("Tool requested termination - ending supervisor execution")
                        from langgraph.types import Command
                        return Command(goto=END, update=result)
        
        # Extract structured response
        structured_response = None
        response_content = "Task analysis completed"
        
        if result and "structured_response" in result:
            structured_response = result["structured_response"]
            response_content = structured_response.message
            
            # Update supervisor status based on structured response
            if structured_response.status in ["analyzing", "planning", "delegating", "monitoring", "completed", "failed"]:
                supervisor_state.supervisor_status = structured_response.status
                
        elif result and "messages" in result:
            # Fallback to message content if no structured response
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_content = last_message.content
            else:
                response_content = str(last_message)
        
        # Check if supervisor status indicates completion - if so, terminate
        if supervisor_state.supervisor_status in ["completed", "failed"]:
            logger.info(f"Supervisor status is {supervisor_state.supervisor_status} - terminating execution")
            from langgraph.types import Command
            return Command(
                goto=END,
                update={
                    "current_reasoning": response_content,
                    "supervisor_status": supervisor_state.supervisor_status,
                    "updated_at": datetime.now(),
                    "structured_response": structured_response
                }
            )
        
        # Update state
        supervisor_state.current_reasoning = response_content
        supervisor_state.updated_at = datetime.now()
        
        # Return both regular state updates and structured response for external use
        return {
            "current_reasoning": response_content,
            "supervisor_status": supervisor_state.supervisor_status,
            "updated_at": supervisor_state.updated_at,
            "structured_response": structured_response
        }
        
    except Exception as e:
        logger.error(f"Error in top level supervisor: {e}")
        # Handle case where supervisor_state creation failed
        if supervisor_state is None:
            if isinstance(state, dict):
                error_messages = state.get("error_messages", []) + [f"Supervisor error: {str(e)}"]
            else:
                error_messages = [f"Supervisor error: {str(e)}"]
            
            return {
                "error_messages": error_messages,
                "supervisor_status": "failed",
                "updated_at": datetime.now()
            }
        else:
            supervisor_state.error_messages.append(f"Supervisor error: {str(e)}")
            supervisor_state.supervisor_status = "failed"
            supervisor_state.updated_at = datetime.now()
            
            return {
                "error_messages": supervisor_state.error_messages,
                "supervisor_status": "failed",
                "updated_at": supervisor_state.updated_at
            }