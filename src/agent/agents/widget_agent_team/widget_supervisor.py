"""Widget supervisor node with intelligent routing based on complete state analysis."""

import json
from datetime import datetime
from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langgraph.graph import END
from langgraph.types import Command

from agent.models import SupervisorDecision, WidgetAgentState


class WidgetSupervisor:
    """Intelligent supervisor that analyzes complete state and routes to appropriate nodes."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize the supervisor with LLM."""
        self.llm = init_chat_model(llm_model)
        self.llm_with_structure = self.llm.with_structured_output(SupervisorDecision)

    def analyze_state(self, state: WidgetAgentState) -> Dict[str, Any]:
        """Analyze the current state comprehensively."""
        return {
            "task_progress": {
                "has_raw_data": state.raw_file_data is not None,
                "has_generated_code": state.generated_code is not None,
                "has_execution_result": state.code_execution_result is not None,
                "is_validated": state.data_validated,
                "has_errors": len(state.error_messages) > 0,
                "iteration_count": state.iteration_count,
                "current_status": state.task_status,
            },
            "task_context": {
                "task_id": state.task_id,
                "widget_type": state.widget_type,
                "operation": state.operation,
                "user_prompt": state.user_prompt,
                "task_instructions": state.task_instructions,
                "file_count": len(state.file_ids),
            },
            "error_context": {
                "recent_errors": state.error_messages[-3:]
                if state.error_messages
                else [],
                "total_error_count": len(state.error_messages),
            },
            "data_context": {
                "has_file_schemas": len(state.file_schemas) > 0,
                "has_sample_data": len(state.file_sample_data) > 0,
                "widget_configured": bool(state.title and state.description),
            },
        }

    def create_routing_prompt(self, state_analysis: Dict[str, Any]) -> str:
        """Create comprehensive routing prompt for LLM supervisor."""
        return f"""
You are an intelligent supervisor managing a widget data processing pipeline.

CURRENT STATE ANALYSIS:
{json.dumps(state_analysis, indent=2)}

AVAILABLE NODES:
1. data: Unified node that fetches data if needed, generates code, and executes it (run if data processing is needed)
2. validate_data: Validates the execution result matches widget requirements (with LLM-based confidence scoring)
3. end: Complete the workflow

DECISION RULES:
- If data processing is needed (no raw data, no code, or no execution result), use 'data' node
- If execution result exists but not validated, use 'validate_data' node  
- If data is validated successfully (confidence >= 80), use 'end' to complete the task
- If validation failed and iteration_count < 3, retry with 'data' node (validation provides feedback for retry)
- If iteration_count >= 3, use 'end' with failure status
- The validate_data node now handles task completion internally when validation passes

Make an intelligent routing decision based on the current state. Consider:
- The logical flow of the pipeline
- Error recovery strategies
- Maximum retry limits
- Task completion conditions

Return your decision with clear reasoning.
"""

    def apply_business_rules(
        self, decision: SupervisorDecision, state: WidgetAgentState
    ) -> SupervisorDecision:
        """Apply business logic constraints to the decision."""

        # Max retry constraint
        if state.iteration_count >= 3 and decision.next_node == "data":
            decision.next_node = "end"
            decision.reasoning = (
                "Maximum retry attempts reached, ending task with failure"
            )

        # Missing required fields for data processing
        if decision.next_node == "data" and not state.user_prompt:
            decision.next_node = "end"
            decision.reasoning = (
                "Missing user prompt required for data processing, ending with failure"
            )

        return decision

    def __call__(self, state: WidgetAgentState) -> Command:
        """
        Main supervisor function that reads complete state and makes intelligent routing decisions.
        """
        try:
            # EXPLICIT COMPLETION CHECKS - Handle definitive completion conditions first

            # 1. Task explicitly marked as completed
            if state.task_status == "completed":
                return Command(
                    goto=END,
                    update={
                        "widget_supervisor_reasoning": "Task marked as completed by validation",
                        "updated_at": datetime.now(),
                    },
                )

            # 2. Task explicitly marked as failed
            if state.task_status == "failed":
                return Command(
                    goto=END,
                    update={
                        "widget_supervisor_reasoning": "Task marked as failed",
                        "updated_at": datetime.now(),
                    },
                )

            # 3. Max iterations reached - force completion
            if state.iteration_count >= 3:
                final_status = "completed" if state.data_validated else "failed"
                return Command(
                    goto=END,
                    update={
                        "widget_supervisor_reasoning": "Maximum iterations reached, ending task",
                        "task_status": final_status,
                        "updated_at": datetime.now(),
                    },
                )

            # If no explicit completion condition, proceed with LLM routing
            # Comprehensive state analysis
            state_analysis = self.analyze_state(state)

            # Create detailed prompt for LLM supervisor
            routing_prompt = self.create_routing_prompt(state_analysis)

            # Get structured decision from LLM
            decision = self.llm_with_structure.invoke(routing_prompt)

            # Apply business logic constraints
            decision = self.apply_business_rules(decision, state)

            # Handle end condition
            if decision.next_node == "end":
                final_status = "completed" if state.data_validated else "failed"
                return Command(
                    goto=END,
                    update={
                        "widget_supervisor_reasoning": decision.reasoning,
                        "task_status": final_status,
                        "updated_at": datetime.now(),
                    },
                )

            # Return routing command with state updates
            return Command(
                goto=decision.next_node,
                update={
                    "current_step": decision.next_node,
                    "widget_supervisor_reasoning": decision.reasoning,
                    "updated_at": datetime.now(),
                },
            )

        except Exception as e:
            # Handle supervisor errors gracefully
            error_msg = f"Supervisor error: {str(e)}"
            return Command(
                goto=END,
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "widget_supervisor_reasoning": f"Supervisor failed: {error_msg}",
                    "task_status": "failed",
                    "updated_at": datetime.now(),
                },
            )


# Create lazy singleton instance for graph usage
_widget_supervisor_instance = None


def widget_supervisor(state: WidgetAgentState) -> Command:
    """Lazy singleton wrapper for WidgetSupervisor."""
    global _widget_supervisor_instance
    if _widget_supervisor_instance is None:
        _widget_supervisor_instance = WidgetSupervisor()
    return _widget_supervisor_instance(state)
