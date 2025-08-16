"""Data processing agent for widget creation."""

from datetime import datetime
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from agent.models import FileSampleData, FileSchema, WidgetAgentState
from .tools.fetch_data import fetch_data_tool
from .tools.code_generation import generate_python_code_tool
from .tools.code_execution import e2b_sandbox_tool


class DataAgent:
    """Data processing agent using create_react_agent."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize data agent with LLM."""
        self.llm_model = llm_model

        # Create extended state schema that includes required AgentState fields
        # Define reducers for fields that might be updated concurrently
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

        class ExtendedWidgetState(TypedDict):
            # Required AgentState fields for create_react_agent
            messages: Annotated[Sequence[BaseMessage], add_messages]
            remaining_steps: int

            # Custom WidgetAgentState fields (with reducers for concurrent updates)
            task_id: str
            task_status: Annotated[Optional[str], take_last]
            task_instructions: str
            user_prompt: str
            operation: str
            widget_type: str
            widget_id: str
            dashboard_id: str
            file_ids: list[str]
            file_sample_data: Annotated[Optional[list[FileSampleData]], take_last]
            file_schemas: Annotated[Optional[list[FileSchema]], take_last]
            title: str
            description: str
            data: Annotated[Optional[dict], take_last]
            widget_metadata: Annotated[Optional[dict], take_last]
            data_validated: Annotated[Optional[bool], take_last]
            raw_file_data: Annotated[Optional[dict], take_last]
            generated_code: Annotated[Optional[str], take_last]
            code_execution_result: Annotated[Optional[dict], take_last]
            error_messages: Annotated[list[str], merge_lists]
            iteration_count: Annotated[Optional[int], take_last]
            current_step: Annotated[Optional[str], take_last]
            widget_supervisor_reasoning: Annotated[Optional[str], take_last]
            # Database operation completion flags
            widget_creation_completed: Annotated[Optional[bool], take_last]
            widget_update_completed: Annotated[Optional[bool], take_last]
            widget_deletion_completed: Annotated[Optional[bool], take_last]
            created_at: datetime
            updated_at: Annotated[Optional[datetime], take_last]

        # Create the main data processing agent with extended state schema
        self.agent = create_react_agent(
            model=llm_model,
            tools=[fetch_data_tool, generate_python_code_tool, e2b_sandbox_tool],
            state_schema=ExtendedWidgetState,
            prompt="""You are a data processing agent for widget creation. Your job is to:
            
1. First, use fetch_data_tool (no parameters needed - extracts file_ids from state)
2. Then, use generate_python_code_tool (no parameters needed - extracts requirements from state) 
3. Finally, use e2b_sandbox_tool (no parameters needed - executes generated code from state)

Always follow this sequence: fetch data → generate code → execute code.
Make sure the final result matches the required format for the widget type.
            """,
        )

        # Create code generation sub-agent
        self.code_generator_agent = create_react_agent(
            model=llm_model,
            tools=[generate_python_code_tool],
            prompt="""You are a specialized Python code generator for data visualization widgets. 
            Generate clean, efficient Python code that transforms raw data into the exact format required for each widget type.
            Focus on data manipulation using pandas and numpy when needed.
            Always set your final result in a variable called 'result'.
            """,
        )

    def process_data(self, state: WidgetAgentState) -> Command:
        """Unified data processing using create_react_agent with proper state handling."""
        try:
            # Create initial message for the agent
            initial_message = f"""Process data for {state.widget_type} widget:
            
User request: {state.user_prompt}
Task instructions: {state.task_instructions}
Operation: {state.operation}
File IDs: {state.file_ids}
            
Please:
1. Use fetch_data_tool (no parameters needed - extracts file_ids from state)
2. Use generate_python_code_tool (no parameters needed - extracts requirements from state)
3. Use e2b_sandbox_tool (no parameters needed - executes generated code from state)
            """

            # Convert WidgetAgentState to ExtendedWidgetState format
            agent_input = {
                "messages": [{"role": "user", "content": initial_message}],
                "remaining_steps": 10,  # Default step limit
                "task_id": state.task_id,
                "task_status": state.task_status,
                "task_instructions": state.task_instructions,
                "user_prompt": state.user_prompt,
                "operation": state.operation,
                "widget_type": state.widget_type,
                "widget_id": state.widget_id,
                "dashboard_id": state.dashboard_id,
                "file_ids": state.file_ids,
                "file_sample_data": state.file_sample_data or [],
                "file_schemas": state.file_schemas or [],
                "title": state.title,
                "description": state.description,
                "data": state.data,
                "widget_metadata": state.widget_metadata,
                "data_validated": state.data_validated,
                "raw_file_data": state.raw_file_data,
                "generated_code": state.generated_code,
                "code_execution_result": state.code_execution_result,
                "error_messages": state.error_messages or [],
                "iteration_count": state.iteration_count,
                "current_step": state.current_step,
                "widget_supervisor_reasoning": state.widget_supervisor_reasoning,
                # Database operation completion flags
                "widget_creation_completed": state.widget_creation_completed,
                "widget_update_completed": state.widget_update_completed,
                "widget_deletion_completed": state.widget_deletion_completed,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
            }

            # Invoke the create_react_agent - it will handle tool calling with state injection
            agent_result = self.agent.invoke(agent_input)

            # The agent result contains the updated state from Command objects
            # Extract what we need for the parent graph
            update_dict = {
                "task_status": "in_progress",
                "updated_at": datetime.now(),
                "iteration_count": state.iteration_count + 1,
            }

            # Extract updated fields from agent result
            # Only extract fields that were actually updated by the tools
            if (
                "raw_file_data" in agent_result
                and agent_result["raw_file_data"] is not None
            ):
                update_dict["raw_file_data"] = agent_result["raw_file_data"]
            if (
                "file_schemas" in agent_result
                and agent_result["file_schemas"] is not None
            ):
                update_dict["file_schemas"] = agent_result["file_schemas"]
            if (
                "file_sample_data" in agent_result
                and agent_result["file_sample_data"] is not None
            ):
                update_dict["file_sample_data"] = agent_result["file_sample_data"]
            if (
                "generated_code" in agent_result
                and agent_result["generated_code"] is not None
            ):
                update_dict["generated_code"] = agent_result["generated_code"]
            if (
                "code_execution_result" in agent_result
                and agent_result["code_execution_result"] is not None
            ):
                update_dict["code_execution_result"] = agent_result[
                    "code_execution_result"
                ]
            if (
                "error_messages" in agent_result
                and agent_result["error_messages"] is not None
            ):
                # For error_messages, we want to merge with existing ones
                existing_errors = state.error_messages or []
                new_errors = agent_result["error_messages"]
                if isinstance(new_errors, list):
                    all_errors = existing_errors + new_errors
                else:
                    all_errors = existing_errors + [new_errors]
                update_dict["error_messages"] = list(
                    set(all_errors)
                )  # Remove duplicates

                # Check if there's an error in the execution result
                if (
                    isinstance(agent_result.get("code_execution_result"), dict)
                    and "error" in agent_result["code_execution_result"]
                ):
                    error_msg = f"Code execution returned error: {agent_result['code_execution_result']['error']}"
                    if error_msg not in update_dict["error_messages"]:
                        update_dict["error_messages"].append(error_msg)

            return Command(goto="widget_supervisor", update=update_dict)

        except Exception as e:
            return Command(
                goto="widget_supervisor",
                update={
                    "error_messages": state.error_messages
                    + [f"Data node error: {str(e)}"],
                    "updated_at": datetime.now(),
                    "task_status": "failed",  # Only update task_status in error case
                },
            )