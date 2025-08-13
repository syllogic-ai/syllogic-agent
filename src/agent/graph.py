"""LangGraph implementation that imports properly structured agent from coordinator module.

This graph module serves as the main entry point and delegates to the properly
structured top_level_coordinator agent that uses create_react_agent pattern.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from supabase import Client

from agent.agents.top_level_coordinator.top_level_coordinator import top_level_coordinator
from agent.models import TopLevelState


class Context(TypedDict):
    """Context parameters for the chart generation system."""

    openai_api_key: str
    model: str
    supabase_url: str
    supabase_key: str
    supabase_client: Client


async def coordinator_node(
    state: TopLevelState, runtime: Runtime[Context]
) -> TopLevelState:
    """Coordinator node that delegates to the proper agent module."""
    try:
        context = runtime.context or {}

        if not state.user_prompt:
            state.errors.append("No user prompt provided")
            state.should_continue = False
            return state

        # Delegate to the properly structured coordinator agent
        return top_level_coordinator(state, context)

    except Exception as e:
        error_msg = f"Error in coordinator node: {str(e)}"
        state.errors.append(error_msg)
        state.should_continue = False
        state.result = f"Error: {error_msg}"
        return state


# Build the graph following LangGraph best practices
builder = StateGraph(TopLevelState, context_schema=Context)

# Add single coordinator node
builder.add_node("coordinator", coordinator_node)

# Add edges - direct flow from start to coordinator to end
builder.add_edge(START, "coordinator")
builder.add_edge("coordinator", END)

# Compile the graph
graph = builder.compile(name="Chart Generation System")


# Export for langgraph server
__all__ = ["graph"]
