"""Widget Agent Team - A LangGraph-based system for processing data visualization tasks.

This module implements a supervisor pattern where a central widget_supervisor node
intelligently routes tasks to specialized worker nodes for widget processing.

Key Components:
- WidgetSupervisor: Central coordinator that analyzes state and routes tasks
- Worker Nodes: Specialized nodes for data fetching, code generation, execution, validation
- Tools: Helper functions for data processing and code generation
- Main: Entry point and usage examples

Usage:
    from agent.agents.widget_agent_team import WidgetAgentRunner, create_custom_widget

    runner = WidgetAgentRunner()
    task = runner.create_sample_task(
        user_prompt="Create a bar chart showing sales data",
        widget_type="bar",
        file_ids=["file123"]
    )
    result = await runner.process_widget_task(task)
"""

from .main import WidgetAgentRunner, create_custom_widget, run_all_examples
from .tools import CodeGenerator, DataProcessor
from .widget_supervisor import WidgetSupervisor, widget_supervisor
from .worker_nodes import (
    data_node,
    validate_data_node,
)

__all__ = [
    # Main components
    "WidgetAgentRunner",
    "create_custom_widget",
    "run_all_examples",
    # Supervisor
    "WidgetSupervisor",
    "widget_supervisor",
    # Worker nodes
    "data_node",
    "validate_data_node",
    # Tools
    "DataProcessor",
    "CodeGenerator",
]
