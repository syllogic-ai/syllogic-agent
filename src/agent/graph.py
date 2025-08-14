"""LangGraph implementation for the Widget Agent System.

This graph implements a supervisor pattern where a widget_supervisor node
intelligently routes tasks to specialized worker nodes for widget processing.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from supabase import Client

from agent.agents.widget_agent_team.widget_supervisor import widget_supervisor
from agent.agents.widget_agent_team.worker_nodes import (
    data_node,
    update_task_node,
    validate_data_node,
)
from agent.models import WidgetAgentState


class Context(TypedDict):
    """Context parameters for the widget agent system."""

    openai_api_key: str
    model: str
    supabase_url: str
    supabase_key: str  # Can be either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    supabase_client: Client


def build_widget_agent_graph():
    """Builds and compiles the complete widget agent graph."""

    # Initialize the graph with Pydantic state
    builder = StateGraph(WidgetAgentState)

    # Add the widget_supervisor node
    builder.add_node("widget_supervisor", widget_supervisor)

    # Add all worker nodes
    builder.add_node("data", data_node)
    builder.add_node("validate_data", validate_data_node)
    builder.add_node("update_task", update_task_node)

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
