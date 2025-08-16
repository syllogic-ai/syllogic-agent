"""LangGraph implementation for the Multi-Agent System.

This graph implements a hierarchical supervisor pattern where a top_level_supervisor 
orchestrates tasks across specialized agent teams, including the widget_agent_team
for widget processing operations.
"""

from __future__ import annotations

import os

# Import reducers directly from utils file to avoid dependency issues
import sys
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from supabase import Client

from agent.agents.widget_agent_team.widget_supervisor import widget_supervisor
from agent.agents.widget_agent_team.worker_nodes import (
    data_node,
    validate_data_node,
    db_operations_node,
)
from agent.agents.top_level_supervisor.top_level_supervisor import top_level_supervisor
from agent.models import WidgetAgentState, TopLevelSupervisorState

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


def build_top_level_supervisor_graph():
    """Builds and compiles the complete multi-agent system with top-level supervisor."""

    # Use TopLevelSupervisorState with LangGraph requirements
    class SupervisorGraphState(TopLevelSupervisorState):
        # Add required LangGraph fields
        messages: Annotated[Sequence[BaseMessage], add_messages]
        remaining_steps: int

    # Initialize the graph with TopLevelSupervisorState
    builder = StateGraph(SupervisorGraphState)

    # Add the top-level supervisor node
    builder.add_node("top_level_supervisor", top_level_supervisor)

    # Define edges - START goes to top_level_supervisor
    builder.add_edge(START, "top_level_supervisor")

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
    "build_complete_agent_system"
]
