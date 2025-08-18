"""LangGraph implementation for the Multi-Agent System.

This graph implements a hierarchical supervisor pattern where a top_level_supervisor 
orchestrates tasks across specialized agent teams, including the widget_agent_team
for widget processing operations.
"""

from __future__ import annotations

import os

# Import reducers directly from utils file to avoid dependency issues
import sys
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from supabase import Client

from agent.agents.widget_agent_team.widget_supervisor import widget_supervisor
from agent.agents.widget_agent_team.worker_nodes import (
    data_node,
    validate_data_node,
    db_operations_node,
)
from agent.agents.top_level_supervisor.top_level_supervisor import top_level_supervisor
from agent.models import WidgetAgentState, TopLevelSupervisorState, DelegatedTask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../actions"))

from utils import merge_lists, take_last


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
    class WidgetGraphState(WidgetAgentState):
        # Add required LangGraph fields
        messages: Annotated[Sequence[BaseMessage], add_messages]
        remaining_steps: int

    # Initialize the graph with WidgetAgentState
    builder = StateGraph(WidgetGraphState)

    # Add the widget_supervisor node
    builder.add_node("widget_supervisor", widget_supervisor)

    # Add all worker nodes
    builder.add_node("data", data_node)
    builder.add_node("validate_data", validate_data_node)
    builder.add_node("db_operations_node", db_operations_node)

    # Define edges - START goes to widget_supervisor
    builder.add_edge(START, "widget_supervisor")

    # Worker nodes return to supervisor for continued routing
    builder.add_edge("data", "widget_supervisor")
    builder.add_edge("validate_data", "widget_supervisor")
    builder.add_edge("db_operations_node", "widget_supervisor")

    # Compile the graph
    widget_graph = builder.compile(name="Widget Agent System")

    return widget_graph


def call_widget_agent_team(state: dict):
    """Node function to call the widget agent team subgraph.
    
    This function serves as a proper subgraph integration between the top-level supervisor
    and the widget agent team, using the actual widget_agent_graph as a subgraph.
    """
    from langgraph.types import Command
    
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
        
        # Convert state to WidgetAgentState format for the subgraph
        widget_state = {
            "messages": [],
            "remaining_steps": 10,
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
        
        # Call the widget agent subgraph
        widget_result = widget_agent_graph.invoke(widget_state)
        
        # Update the task status based on subgraph result
        task_status = widget_result.get("task_status", "failed")
        if task_status == "completed":
            current_task["task_status"] = "completed"
            current_task["result"] = "Widget operation completed successfully"
        else:
            current_task["task_status"] = "failed"
            current_task["result"] = "Widget operation failed"
        
        # Update the main state
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
    
    # Check if supervisor wants to delegate tasks
    current_reasoning = get_state_value(state, 'current_reasoning')
    if current_reasoning and "DELEGATE_TO_WIDGET_TEAM" in current_reasoning:
        return "widget_agent_team"
    
    # Check if there are pending widget tasks
    delegated_tasks = get_state_value(state, 'delegated_tasks', [])
    pending_widget_tasks = [
        task for task in delegated_tasks 
        if (isinstance(task, dict) and task.get("target_agent") == "widget_agent_team" and task.get("task_status") == "pending")
        or (hasattr(task, 'target_agent') and task.target_agent == "widget_agent_team" and task.task_status == "pending")
    ]
    
    if pending_widget_tasks:
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
    
    # Add widget agent team as a callable node
    builder.add_node("widget_agent_team", call_widget_agent_team)

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
