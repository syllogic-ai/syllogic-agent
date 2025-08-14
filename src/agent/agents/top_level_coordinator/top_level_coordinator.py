"""Top Level Coordinator - orchestrates the entire chart generation workflow using create_react_agent.

This coordinator uses LangGraph's create_react_agent pattern with tools to:
1. Fetch dashboard files
2. Get schemas and sample data
3. Analyze data sufficiency against user request
4. Create widget tasks based on analysis
5. Provide clear workflow coordination
"""

import logging
from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from actions.dashboard import (
    get_files_from_dashboard,
    get_sample_data_from_files,
    get_schemas_from_files,
)
from agent.models import TopLevelState

logger = logging.getLogger(__name__)


def _initialize_dashboard_context(
    state: TopLevelState, runtime_context: dict
) -> TopLevelState:
    """Initialize state with dashboard context by loading files, schemas, and sample data."""
    try:
        supabase = runtime_context["supabase_client"]

        # Get files from dashboard
        file_ids = get_files_from_dashboard(supabase, state.dashboard_id)

        # Get schemas and sample data for all files
        schemas = get_schemas_from_files(supabase, file_ids)
        samples = get_sample_data_from_files(supabase, file_ids)

        # Update state with dashboard context
        state.file_ids = file_ids
        state.available_data_schemas = schemas
        state.available_sample_data = samples

        logger.info(
            f"Loaded dashboard context: {len(file_ids)} files, {len(schemas)} schemas, {len(samples)} samples"
        )
        return state

    except Exception as e:
        error_msg = f"Error initializing dashboard context: {str(e)}"
        state.errors.append(error_msg)
        logger.error(error_msg, exc_info=True)
        return state


# Collect all tools for the coordinator
coordinator_tools = []


def create_coordinator_agent(runtime_context: dict) -> Any:
    """Create a ReAct agent with all necessary tools."""
    llm = ChatOpenAI(
        model=runtime_context.get("model", "gpt-5-mini"),
        temperature=0.1,
        api_key=runtime_context.get("openai_api_key"),
    )

    # According to LangGraph documentation, create_react_agent takes model and tools
    # The system message should be added to the prompt directly or via state_schema
    return create_react_agent(llm, coordinator_tools)


def top_level_coordinator(state: TopLevelState, runtime_context: dict) -> TopLevelState:
    """Coordinator that uses ReAct agent with tools to orchestrate chart generation workflow.

    Args:
        state: Current chart generation state
        runtime_context: Runtime context with configuration

    Returns:
        Updated TopLevelState
    """
    try:
        logger.info(f"Starting top level coordinator for request {state.request_id}")

        # Initialize state with dashboard context if not already done
        if (
            not state.file_ids
            and state.dashboard_id
            and runtime_context.get("supabase_client")
        ):
            logger.info(
                f"Initializing dashboard context for dashboard {state.dashboard_id}"
            )
            state = _initialize_dashboard_context(state, runtime_context)

        if not state.started_at:
            state.started_at = datetime.now()

        if not state.user_prompt:
            state.errors.append("No user prompt provided")
            state.should_continue = False
            return state

        # Create the ReAct agent with tools
        agent = create_coordinator_agent(runtime_context)

        # Prepare the input for the agent with system message
        agent_input = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a data analysis coordinator for a dashboard system.

Your job is to help users create charts and widgets based on their requests.

Workflow:
1. Analyze if the data is sufficient for the user's request
2. If sufficient, create widget tasks with appropriate parameters
3. Provide a clear summary of what you accomplished

Always be helpful and explain what you're doing at each step.""",
                },
                {
                    "role": "user",
                    "content": f"""
Please help me with this request: "{state.user_prompt}"

Context:
- Dashboard ID: {state.dashboard_id}
- Request ID: {state.request_id}

Please follow this workflow:
1. Analyze if the available data is sufficient for creating: "{state.user_prompt}"
2. If sufficient, create appropriate widget tasks
3. Provide a clear summary of what you accomplished

Please be thorough and use the tools available to you.
""",
                },
            ]
        }

        # Run the agent with tools
        result = agent.invoke(agent_input)

        # Extract the final message from the agent
        if result.get("messages"):
            final_message = result["messages"][-1]
            state.result = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )
        else:
            state.result = "Agent completed processing"

        state.current_step = "completed"
        state.should_continue = False
        state.completed_at = datetime.now()

        logger.info(f"Coordinator completed for request {state.request_id}")
        return state

    except Exception as e:
        error_msg = f"Error in coordinator: {str(e)}"
        state.errors.append(error_msg)
        state.should_continue = False
        state.result = f"Error: {error_msg}"
        logger.error(error_msg, exc_info=True)
        return state


def should_continue_coordinator(state: TopLevelState) -> str:
    """Determine the next step in the workflow based on current state.

    Args:
        state: Current chart generation state

    Returns:
        Next node name or "__end__"
    """
    try:
        # Check for errors first
        if state.errors:
            logger.info("Ending workflow due to errors")
            return "__end__"

        # Check current step
        if state.current_step == "completed":
            logger.info("Coordinator completed - ending workflow")
            return "__end__"

        if state.current_step == "task_processing":
            logger.info("Continuing to task processing")
            return "task_processor"  # Assuming this is the next node

        if state.should_continue:
            logger.info("Continuing workflow based on should_continue flag")
            return "task_processor"  # Default next step

        logger.info("No clear next step - ending workflow")
        return "__end__"

    except Exception as e:
        logger.error(f"Error in routing decision: {str(e)}")
        return "__end__"
