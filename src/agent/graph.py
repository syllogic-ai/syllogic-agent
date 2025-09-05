"""LangGraph implementation for the Multi-Agent System.

This graph implements a hierarchical supervisor pattern where a top_level_supervisor 
orchestrates tasks across specialized agent teams. The widget_agent_team is implemented
as a subgraph that can be called from the main graph.

Following CLAUDE.md guidelines, this file contains ONLY graph structure definition.
All business logic has been moved to appropriate helper modules in src/actions/.
"""

from __future__ import annotations

import os

# Import reducers directly from utils file to avoid dependency issues
import sys
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, List, Optional

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from supabase import Client

# Import graph helper functions from actions
from actions.graph_helpers import (
    should_delegate_to_widget_team,
    convert_backend_payload_to_graph_state,
    widget_team_adapter
)

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
    builder.add_node("db_operations", db_operations_node)
    builder.add_node("text_block_node", text_block_node)

    # Define entry point - START goes to widget_supervisor
    builder.add_edge(START, "widget_supervisor")

    # No hardcoded edges between worker nodes and supervisor
    # Worker nodes use Command objects to dynamically route back to supervisor
    # This follows LangGraph best practices where nodes control their own routing

    # Compile the graph
    widget_graph = builder.compile(name="Widget Agent System")

    return widget_graph




# Business logic functions have been moved to actions.graph_helpers
# This follows CLAUDE.md guidelines: graph.py should contain ONLY graph structure


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
    # Using helper function from actions.graph_helpers (follows CLAUDE.md guidelines)
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
