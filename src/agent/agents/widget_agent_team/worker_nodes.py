"""Worker nodes for widget data processing pipeline using create_react_agent."""

from langgraph.types import Command

from agent.models import WidgetAgentState
from .data_agent import DataAgent
from .validation_agent import ValidationAgent
from .database_agent import DatabaseAgent
from .tools.fetch_data import fetch_data_tool
from .tools.code_generation import generate_python_code_tool
from .tools.code_execution import e2b_sandbox_tool




class WorkerNodes:
    """Collection of worker nodes for widget processing using create_react_agent."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize worker nodes with specialized agents."""
        self.llm_model = llm_model
        self.data_agent = DataAgent(llm_model)
        self.validation_agent = ValidationAgent(llm_model)
        self.database_agent = DatabaseAgent()

    def data_node(self, state: WidgetAgentState) -> Command:
        """Unified data processing node using create_react_agent with proper state handling."""
        return self.data_agent.process_data(state)

    def validate_data_node(self, state: WidgetAgentState) -> Command:
        """LLM-based validation that analyzes generated data against user requirements."""
        return self.validation_agent.validate_data(state)

    def db_operations_node(self, state: WidgetAgentState) -> Command:
        """
        Database operations node that handles CREATE/UPDATE/DELETE operations for widgets.
        Uses create_widget, update_widget, delete_widget from dashboard.py.
        """
        return self.database_agent.perform_db_operations(state)


# Create lazy singleton instance
_worker_nodes_instance = None


def get_worker_nodes():
    """Get or create worker nodes instance."""
    global _worker_nodes_instance
    if _worker_nodes_instance is None:
        _worker_nodes_instance = WorkerNodes()
    return _worker_nodes_instance


# Export individual node functions for graph usage with lazy initialization
def data_node(state: WidgetAgentState) -> Command:
    """Lazy wrapper for data_node."""
    return get_worker_nodes().data_node(state)


def validate_data_node(state: WidgetAgentState) -> Command:
    """Lazy wrapper for validate_data_node."""
    return get_worker_nodes().validate_data_node(state)


def db_operations_node(state: WidgetAgentState) -> Command:
    """Lazy wrapper for db_operations_node."""
    return get_worker_nodes().db_operations_node(state)


# Export tools and nodes
__all__ = [
    "fetch_data_tool",
    "e2b_sandbox_tool", 
    "generate_python_code_tool",
    "data_node",
    "validate_data_node",
    "db_operations_node",
    "get_worker_nodes",
    "WorkerNodes",
]
