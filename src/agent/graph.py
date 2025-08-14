"""LangGraph implementation for the Widget Agent System.

This graph implements a supervisor pattern where a widget_supervisor node
intelligently routes tasks to specialized worker nodes for widget processing.
"""

from __future__ import annotations

from typing import TypedDict, Optional, List, Dict, Any, Annotated, Sequence, Literal
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from supabase import Client

from agent.agents.widget_agent_team.widget_supervisor import widget_supervisor
from agent.agents.widget_agent_team.worker_nodes import (
    data_node,
    update_task_node,
    validate_data_node,
)

def convert_to_widget_state(graph_state) -> WidgetAgentState:
    """Convert GraphWidgetState to WidgetAgentState for node functions."""
    from datetime import datetime
    
    # Create a WidgetAgentState from the graph state
    return WidgetAgentState(
        task_id=graph_state.get("task_id", ""),
        task_status=graph_state.get("task_status", "pending"),
        task_instructions=graph_state.get("task_instructions", ""),
        user_prompt=graph_state.get("user_prompt", ""),
        operation=graph_state.get("operation", "CREATE"),
        widget_type=graph_state.get("widget_type", "line"),
        widget_id=graph_state.get("widget_id", ""),
        file_ids=graph_state.get("file_ids", []),
        file_sample_data=graph_state.get("file_sample_data", []),
        file_schemas=graph_state.get("file_schemas", []),
        title=graph_state.get("title", ""),
        description=graph_state.get("description", ""),
        data=graph_state.get("data"),
        widget_metadata=graph_state.get("widget_metadata"),
        data_validated=graph_state.get("data_validated", False),
        raw_file_data=graph_state.get("raw_file_data"),
        generated_code=graph_state.get("generated_code"),
        code_execution_result=graph_state.get("code_execution_result"),
        error_messages=graph_state.get("error_messages", []),
        iteration_count=graph_state.get("iteration_count", 0),
        current_step=graph_state.get("current_step"),
        widget_supervisor_reasoning=graph_state.get("widget_supervisor_reasoning"),
        created_at=graph_state.get("created_at", datetime.now()),
        updated_at=graph_state.get("updated_at")
    )

def wrap_node(node_func):
    """Wrapper to convert state for node functions."""
    def wrapped_node(state):
        widget_state = convert_to_widget_state(state)
        return node_func(widget_state)
    return wrapped_node
from agent.models import WidgetAgentState, FileSchema, FileSampleData

# Define reducers for concurrent updates at module level
def take_last(old, new):
    """Reducer that takes the last (most recent) value"""
    return new if new is not None else old
    
def merge_lists(old, new):
    """Reducer that extends lists with new items"""
    if old is None:
        old = []
    if new is None:
        return old
    if isinstance(new, list):
        return old + new
    return old + [new]


class Context(TypedDict):
    """Context parameters for the widget agent system."""

    openai_api_key: str
    model: str
    supabase_url: str
    supabase_key: str  # Can be either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    supabase_client: Client


def build_widget_agent_graph():
    """Builds and compiles the complete widget agent graph."""

    
    # Create extended state schema for the parent graph
    class GraphWidgetState(TypedDict):
        # Required AgentState fields for tools
        messages: Annotated[Sequence[BaseMessage], add_messages]
        remaining_steps: int
        
        # Custom WidgetAgentState fields (with reducers for concurrent updates)
        task_id: str
        task_status: Annotated[Optional[str], take_last]
        task_instructions: str
        user_prompt: str
        operation: Literal["CREATE", "UPDATE", "DELETE"]
        widget_type: Literal["line", "bar", "pie", "area", "radial", "kpi", "table"]
        widget_id: str
        file_ids: List[str]
        file_sample_data: Annotated[Optional[List[FileSampleData]], take_last]
        file_schemas: Annotated[Optional[List[FileSchema]], take_last]
        title: str
        description: str
        data: Annotated[Optional[Dict[str, Any]], take_last]
        widget_metadata: Annotated[Optional[Dict[str, Any]], take_last]
        data_validated: Annotated[Optional[bool], take_last]
        raw_file_data: Annotated[Optional[Dict[str, Any]], take_last]
        generated_code: Annotated[Optional[str], take_last]
        code_execution_result: Annotated[Optional[Any], take_last]
        error_messages: Annotated[List[str], merge_lists]
        iteration_count: Annotated[Optional[int], take_last]
        current_step: Annotated[Optional[str], take_last]
        widget_supervisor_reasoning: Annotated[Optional[str], take_last]
        created_at: datetime
        updated_at: Annotated[Optional[datetime], take_last]

    # Initialize the graph with the extended state schema
    builder = StateGraph(GraphWidgetState)

    # Add the widget_supervisor node
    builder.add_node("widget_supervisor", wrap_node(widget_supervisor))

    # Add all worker nodes
    builder.add_node("data", wrap_node(data_node))
    builder.add_node("validate_data", wrap_node(validate_data_node))
    builder.add_node("update_task", wrap_node(update_task_node))

    # Define edges - START goes to widget_supervisor
    builder.add_edge(START, "widget_supervisor")

    # All worker nodes report back to widget_supervisor
    # The supervisor uses Command objects to route dynamically
    builder.add_edge("widget_supervisor", "data")
    builder.add_edge("widget_supervisor", "validate_data")
    builder.add_edge("widget_supervisor", "update_task")
    builder.add_edge("widget_supervisor", END)

    # Compile the graph (LangGraph Cloud handles persistence automatically)
    graph = builder.compile(name="Widget Agent System")

    return graph


# Build and export the graph
graph = build_widget_agent_graph()

# Export for langgraph server
__all__ = ["graph", "build_widget_agent_graph"]
