"""LangGraph implementation for the Widget Agent System.

This graph implements a supervisor pattern where a widget_supervisor node
intelligently routes tasks to specialized worker nodes for widget processing.
"""

from __future__ import annotations

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from supabase import Client

from agent.agents.widget_agent_team.widget_supervisor import widget_supervisor
from agent.agents.widget_agent_team.worker_nodes import (
    data_node,
    validate_data_node,
)
from agent.models import WidgetAgentState
# Import reducers directly from utils file to avoid dependency issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../actions'))

from utils import take_last, merge_lists


class Context(TypedDict):
    """Context parameters for the widget agent system."""

    openai_api_key: str
    model: str
    supabase_url: str
    supabase_key: str  # Can be either SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    supabase_client: Client


def build_widget_agent_graph():
    """Builds and compiles the complete widget agent graph."""
    
    # Use WidgetAgentState directly with LangGraph requirements
    class GraphState(WidgetAgentState):
        # Add required LangGraph fields
        messages: Annotated[Sequence[BaseMessage], add_messages]
        remaining_steps: int

    # Initialize the graph with WidgetAgentState
    builder = StateGraph(GraphState)

    # Add the widget_supervisor node
    builder.add_node("widget_supervisor", widget_supervisor)

    # Add all worker nodes
    builder.add_node("data", data_node)
    builder.add_node("validate_data", validate_data_node)

    # Define edges - START goes to widget_supervisor
    builder.add_edge(START, "widget_supervisor")
    
    # Worker nodes return to supervisor for continued routing
    builder.add_edge("data", "widget_supervisor")
    builder.add_edge("validate_data", "widget_supervisor")

    # Compile the graph
    graph = builder.compile(name="Widget Agent System")

    return graph


# Build and export the graph
graph = build_widget_agent_graph()

# Export for langgraph server
__all__ = ["graph", "build_widget_agent_graph"]
