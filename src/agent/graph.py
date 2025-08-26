"""LangGraph implementation for the Multi-Agent System.

This graph implements a hierarchical supervisor pattern where a top_level_supervisor 
orchestrates tasks across specialized agent teams. The widget_agent_team is implemented
as a subgraph that can be called from the main graph.
"""

from __future__ import annotations

import os
import uuid
import logging

# Import reducers directly from utils file to avoid dependency issues
import sys
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, List, Optional

logger = logging.getLogger(__name__)

from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from supabase import Client

from config import get_langchain_config_with_tracing, LANGFUSE_AVAILABLE
from agent.agents.widget_agent_team.widget_supervisor import widget_supervisor
from agent.agents.widget_agent_team.worker_nodes import (
    data_node,
    validate_data_node,
    db_operations_node,
    text_block_node,
)
from agent.agents.top_level_supervisor.top_level_supervisor import top_level_supervisor
from agent.models import WidgetAgentState, TopLevelSupervisorState, DelegatedTask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../actions"))

from utils import merge_lists, take_last


def create_langfuse_config(state: dict, trace_name: str = "syllogic-agent-execution") -> dict:
    """Create Langfuse configuration for tracing based on state context.
    
    Args:
        state: The current graph state containing user/session info
        trace_name: Name for the trace
        
    Returns:
        Dict containing callbacks configuration or empty dict if Langfuse unavailable
    """
    if not LANGFUSE_AVAILABLE:
        return {}
        
    try:
        # Extract context information from state
        user_id = state.get("user_id")
        chat_id = state.get("chat_id") 
        dashboard_id = state.get("dashboard_id")
        request_id = state.get("request_id")
        
        # Create trace metadata
        metadata = {
            "dashboard_id": dashboard_id,
            "request_id": request_id,
        }
        
        # Use the new helper function
        config = get_langchain_config_with_tracing(
            trace_name=trace_name,
            session_id=chat_id,  # Use chat_id as session for conversation grouping
            user_id=user_id,
            tags=["syllogic", "agent", "langgraph", "multi-agent"],
            metadata=metadata
        )
        
        if config:
            logger.info(f"Created Langfuse tracing for user={user_id}, chat={chat_id}, dashboard={dashboard_id}")
            
        return config
            
    except Exception as e:
        logger.error(f"Error creating Langfuse config: {e}")
        return {}


class Context(TypedDict):
    """Context parameters for the widget agent system."""

    openai_api_key: str
    model: str
    supabase_url: str
    supabase_key: str  # Can be either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    supabase_client: Client


def build_widget_agent_graph():
    """Builds and compiles the widget agent subgraph."""

    # Use WidgetAgentState directly with LangGraph requirements
    class WidgetGraphState(TypedDict):
        # LangGraph required fields
        messages: Annotated[Sequence[BaseMessage], add_messages]
        remaining_steps: int
        
        # WidgetAgentState fields (making them all optional for flexibility)
        task_id: Optional[str]
        task_instructions: str
        user_prompt: str
        operation: str  # CREATE, UPDATE, DELETE
        widget_type: str  # line, bar, pie, area, radial, kpi, table
        dashboard_id: str
        chat_id: str
        file_ids: List[str]
        widget_id: Optional[str]
        reference_widget_id: Optional[str]  # Missing field - needed for text block dependencies
        title: Optional[str]
        description: Optional[str]
        task_status: str
        created_at: Optional[datetime]
        updated_at: Optional[datetime]
        started_at: Optional[datetime]
        completed_at: Optional[datetime]
        iteration_count: int
        current_step: Optional[str]
        
        # Widget processing data
        raw_file_data: Optional[dict]
        file_schemas: List[dict]
        file_sample_data: List[dict]
        generated_code: Optional[str]
        code_execution_result: Optional[dict]
        widget_config: Optional[dict]  # Unified config for both text blocks and charts
        data: Optional[dict]
        data_validated: bool
        widget_metadata: Optional[dict]
        error_messages: List[str]
        
        # Database operation flags
        widget_creation_completed: bool
        widget_update_completed: bool
        widget_deletion_completed: bool
        
        # Supervisor reasoning
        widget_supervisor_reasoning: Optional[str]

    # Initialize the graph with WidgetGraphState
    builder = StateGraph(WidgetGraphState)

    # Add the widget_supervisor node with possible destinations
    builder.add_node("widget_supervisor", widget_supervisor)

    # Add worker nodes that can route back to supervisor or END
    # Each worker node uses Command pattern to control its own routing
    builder.add_node("data", data_node)
    builder.add_node("validate_data", validate_data_node)
    builder.add_node("db_operations_node", db_operations_node)
    builder.add_node("text_block_node", text_block_node)

    # Define entry point - START goes to widget_supervisor
    builder.add_edge(START, "widget_supervisor")

    # No hardcoded edges between worker nodes and supervisor
    # Worker nodes use Command objects to dynamically route back to supervisor
    # This follows LangGraph best practices where nodes control their own routing

    # Compile the graph
    widget_graph = builder.compile(name="Widget Agent System")

    return widget_graph




def should_delegate_to_widget_team(state):
    """Conditional edge function to determine if we should delegate to widget team.
    
    This function checks the supervisor's reasoning and state to determine
    if tasks should be delegated to the widget_agent_team.
    """
    # Handle both dict and object state formats
    def get_state_value(state, key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        else:
            return getattr(state, key, default)
    
    # Check if supervisor is completed or failed FIRST (highest priority)
    supervisor_status = get_state_value(state, 'supervisor_status', 'analyzing')
    if supervisor_status in ["completed", "failed"]:
        return "__end__"
    
    # Check if supervisor wants to delegate tasks (look for the delegation signal)
    current_reasoning = get_state_value(state, 'current_reasoning', '')
    if current_reasoning and "DELEGATE_TO_WIDGET_TEAM" in current_reasoning:
        logger.info(f"Delegating to widget team based on reasoning: {current_reasoning}")
        return "widget_agent_team"
    
    # Check if there are pending widget tasks that need processing
    delegated_tasks = get_state_value(state, 'delegated_tasks', [])
    pending_widget_tasks = [
        task for task in delegated_tasks 
        if (isinstance(task, dict) and task.get("target_agent") == "widget_agent_team" and task.get("task_status") == "pending")
        or (hasattr(task, 'target_agent') and task.target_agent == "widget_agent_team" and task.task_status == "pending")
    ]
    
    if pending_widget_tasks:
        logger.info(f"Found {len(pending_widget_tasks)} pending widget tasks - delegating to widget team")
        return "widget_agent_team"
    
    # Continue with supervisor
    return "top_level_supervisor"


def convert_backend_payload_to_graph_state(payload: dict) -> dict:
    """Convert BackendPayload format to SupervisorGraphState format.
    
    This allows the LangGraph API to accept BackendPayload format
    and automatically convert it to the expected graph state.
    """
    from agent.models import BackendPayload
    
    # Validate as BackendPayload first
    validated_payload = BackendPayload(**payload)
    
    # Convert to SupervisorGraphState format
    return {
        "messages": [],
        "remaining_steps": 10,
        "user_prompt": validated_payload.message,  # message -> user_prompt mapping
        "user_id": validated_payload.user_id,
        "dashboard_id": validated_payload.dashboard_id,
        "chat_id": validated_payload.chat_id,
        "request_id": validated_payload.request_id,
        "file_ids": validated_payload.file_ids,
        "context_widget_ids": validated_payload.context_widget_ids or [],
        "available_files": [],
        "available_data_summary": None,
        "delegated_tasks": [],
        "supervisor_status": "analyzing",
        "current_reasoning": None,
        "final_response": None,
        "error_messages": [],
        "created_at": datetime.now(),
        "updated_at": None,
        "all_tasks_completed": False
    }


def call_widget_agent_subgraph(state: dict):
    """Node function to call the widget agent team subgraph with proper state transformation.
    
    This function handles the state transformation between the parent graph's SupervisorGraphState
    and the subgraph's WidgetAgentState, as required by LangGraph when schemas differ.
    """
    try:
        # Get pending widget tasks
        widget_tasks = [task for task in state.get("delegated_tasks", []) 
                       if task.get("target_agent") == "widget_agent_team" and task.get("task_status") == "pending"]
        
        if not widget_tasks:
            return Command(update={
                "error_messages": state.get("error_messages", []) + ["No pending widget tasks found"],
                "current_reasoning": "No widget tasks to process"
            })
        
        # Process the first pending task
        current_task = widget_tasks[0]
        
        # Transform SupervisorGraphState to WidgetAgentState format for the subgraph
        widget_state = {
            "messages": [],  # LangGraph required field
            "remaining_steps": 10,  # LangGraph required field
            "task_instructions": current_task.get("task_instructions", ""),
            "user_prompt": state.get("user_prompt", ""),
            "operation": current_task.get("operation", "CREATE"),
            "widget_type": current_task.get("widget_type", "bar"),
            "dashboard_id": state.get("dashboard_id", ""),
            "chat_id": state.get("chat_id", ""),
            "file_ids": current_task.get("file_ids", []),
            "widget_id": current_task.get("widget_id"),
            "task_status": "in_progress",
            "created_at": datetime.now(),
            "updated_at": None
        }
        
        # Build and invoke the widget agent subgraph
        widget_subgraph = build_widget_agent_graph()
        widget_result = widget_subgraph.invoke(widget_state)
        
        # Transform the subgraph result back to parent state format
        task_status = widget_result.get("task_status", "failed")
        if task_status == "completed":
            current_task["task_status"] = "completed"
            current_task["result"] = "Widget operation completed successfully"
        else:
            current_task["task_status"] = "failed"
            current_task["result"] = "Widget operation failed"
        
        # Update the main state with transformed results
        updated_tasks = state.get("delegated_tasks", [])
        for i, task in enumerate(updated_tasks):
            if task.get("task_id") == current_task.get("task_id"):
                updated_tasks[i] = current_task
                break
        
        return Command(update={
            "delegated_tasks": updated_tasks,
            "current_reasoning": f"Widget subgraph completed: {current_task['result']}"
        })
        
    except Exception as e:
        return Command(update={
            "error_messages": state.get("error_messages", []) + [f"Widget subgraph error: {str(e)}"],
            "current_reasoning": f"Widget subgraph failed: {str(e)}"
        })


def widget_team_adapter(state: dict):
    """Adapter function to handle widget team delegation with proper state transformation."""
    try:
        logger.info("Widget team adapter called - processing delegation")
        
        # Get the first pending widget task
        delegated_tasks = state.get("delegated_tasks", [])
        pending_task = None
        for task in delegated_tasks:
            task_data = task if isinstance(task, dict) else task.__dict__
            if task_data.get("target_agent") == "widget_agent_team" and task_data.get("task_status") == "pending":
                pending_task = task_data
                break
        
        if not pending_task:
            logger.warning("No pending widget task found for delegation")
            # Check if there's widget team input from the delegation
            widget_input = state.get("widget_team_input") or (state.get("widget_team_inputs", [{}])[0])
            if widget_input:
                pending_task = widget_input
            else:
                return Command(update={
                    "current_reasoning": "No pending widget task found",
                    "supervisor_status": "analyzing",
                    # Preserve required fields that must not be lost
                    "user_prompt": state.get("user_prompt", ""),
                    "dashboard_id": state.get("dashboard_id", ""),
                    "user_id": state.get("user_id", f"user_{uuid.uuid4().hex[:8]}"),
                    "chat_id": state.get("chat_id", f"chat_{uuid.uuid4().hex[:8]}"),
                    "request_id": state.get("request_id", f"req_{uuid.uuid4().hex[:8]}"),
                })
        
        # Transform the task data to WidgetAgentState format
        widget_state = {
            "messages": state.get("messages", []),
            "remaining_steps": state.get("remaining_steps", 10),
            "task_id": pending_task.get("task_id", str(uuid.uuid4())),
            "task_instructions": pending_task.get("task_instructions", ""),
            "user_prompt": pending_task.get("user_prompt", state.get("user_prompt", "")),
            "operation": pending_task.get("operation", "CREATE"),
            "widget_type": pending_task.get("widget_type", "bar"),
            "dashboard_id": pending_task.get("dashboard_id", state.get("dashboard_id", "")),
            "chat_id": pending_task.get("chat_id", state.get("chat_id", "")),
            "file_ids": pending_task.get("file_ids", []),
            "widget_id": pending_task.get("widget_id"),
            "title": pending_task.get("title", "Generated Widget"),
            "description": pending_task.get("description", "Widget generated by AI"),
            "task_status": "in_progress",  # Mark as in progress when delegating
            "created_at": datetime.now(),
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
        
        logger.info(f"Invoking widget subgraph for {widget_state['operation']} {widget_state['widget_type']} widget")
        
        # Build and invoke the widget subgraph with Langfuse tracing
        widget_graph = build_widget_agent_graph()
        langfuse_config = create_langfuse_config(state, "widget-subgraph-execution")
        
        if langfuse_config:
            # Invoke with Langfuse tracing
            result = widget_graph.invoke(widget_state, config=langfuse_config)
        else:
            # Invoke without tracing
            result = widget_graph.invoke(widget_state)
        
        # Update the delegated task status in parent state
        updated_tasks = []
        for task in delegated_tasks:
            task_data = task if isinstance(task, dict) else task.__dict__
            if task_data.get("task_id") == pending_task.get("task_id"):
                # Update task status based on widget result
                task_data["task_status"] = result.get("task_status", "completed")
                task_data["completed_at"] = datetime.now()
                if result.get("error_messages"):
                    task_data["error_message"] = "; ".join(result["error_messages"])
                else:
                    task_data["result"] = "Widget operation completed successfully"
            updated_tasks.append(task)
        
        logger.info(f"Widget team completed task with status: {result.get('task_status')}")
        
        # Return Command to update parent state - preserve required fields
        return Command(update={
            "delegated_tasks": updated_tasks,
            "current_reasoning": f"Widget task completed: {result.get('task_status')}",
            "supervisor_status": "analyzing",  # Return to analyzing to check for more tasks
            "updated_at": datetime.now(),
            # Preserve required fields that must not be lost
            "user_prompt": state.get("user_prompt", ""),
            "dashboard_id": state.get("dashboard_id", ""),
            "user_id": state.get("user_id", f"user_{uuid.uuid4().hex[:8]}"),
            "chat_id": state.get("chat_id", f"chat_{uuid.uuid4().hex[:8]}"),
            "request_id": state.get("request_id", f"req_{uuid.uuid4().hex[:8]}"),
        })
        
    except Exception as e:
        logger.error(f"Widget team adapter error: {e}")
        return Command(update={
            "error_messages": state.get("error_messages", []) + [f"Widget team error: {str(e)}"],
            "current_reasoning": f"Widget team failed: {str(e)}",
            "supervisor_status": "failed",
            # Preserve required fields that must not be lost
            "user_prompt": state.get("user_prompt", ""),
            "dashboard_id": state.get("dashboard_id", ""),
            "user_id": state.get("user_id", f"user_{uuid.uuid4().hex[:8]}"),
            "chat_id": state.get("chat_id", f"chat_{uuid.uuid4().hex[:8]}"),
            "request_id": state.get("request_id", f"req_{uuid.uuid4().hex[:8]}"),
        })


def build_top_level_supervisor_graph():
    """Builds and compiles the complete multi-agent system with top-level supervisor."""

    # Create a comprehensive state schema that includes both LangGraph requirements
    # and TopLevelSupervisorState fields
    
    class SupervisorGraphState(TypedDict):
        # LangGraph required fields
        messages: Annotated[Sequence[BaseMessage], add_messages]
        remaining_steps: int
        
        # TopLevelSupervisorState fields - make all optional with defaults
        user_prompt: str
        user_id: str
        dashboard_id: str
        chat_id: str
        request_id: str
        file_ids: List[str]
        context_widget_ids: Optional[List[str]]
        available_files: List[str]
        available_data_summary: Optional[str]
        delegated_tasks: List[DelegatedTask]
        supervisor_status: str
        current_reasoning: Optional[str]
        final_response: Optional[str]
        error_messages: List[str]
        created_at: datetime
        updated_at: Optional[datetime]
        all_tasks_completed: bool

    # Initialize the graph with the comprehensive state
    builder = StateGraph(SupervisorGraphState)

    # Add the top-level supervisor node
    builder.add_node("top_level_supervisor", top_level_supervisor)
    
    # Add widget team adapter as a node to handle Send API delegation
    builder.add_node("widget_agent_team", widget_team_adapter)

    # Define edges - START goes to top_level_supervisor
    builder.add_edge(START, "top_level_supervisor")
    
    # Add conditional edges from supervisor
    builder.add_conditional_edges(
        "top_level_supervisor",
        should_delegate_to_widget_team,
        {
            "widget_agent_team": "widget_agent_team",
            "top_level_supervisor": "top_level_supervisor", 
            "__end__": END
        }
    )
    
    # Widget team returns to supervisor after completion
    builder.add_edge("widget_agent_team", "top_level_supervisor")
    
    # Compile the graph
    supervisor_graph = builder.compile(name="Top Level Supervisor System")

    return supervisor_graph


def build_complete_agent_system():
    """Builds the complete multi-agent system.
    
    This creates both the widget agent subsystem and the top-level supervisor.
    The top-level supervisor orchestrates tasks but does not directly embed
    the widget team - instead it delegates to it through the TopLevelSupervisorRunner.
    """
    
    # Build individual components
    widget_graph = build_widget_agent_graph()
    supervisor_graph = build_top_level_supervisor_graph()
    
    return {
        "widget_agent_system": widget_graph,
        "top_level_supervisor_system": supervisor_graph
    }


# Build and export the graphs
widget_agent_graph = build_widget_agent_graph()
top_level_supervisor_graph = build_top_level_supervisor_graph()
complete_agent_system = build_complete_agent_system()

# Export the top-level supervisor as the main graph for the LangGraph server
graph = top_level_supervisor_graph

# Export for langgraph server and external usage
__all__ = [
    "graph", 
    "widget_agent_graph",
    "top_level_supervisor_graph", 
    "complete_agent_system",
    "build_widget_agent_graph", 
    "build_top_level_supervisor_graph",
    "build_complete_agent_system",
    "convert_backend_payload_to_graph_state"
]
