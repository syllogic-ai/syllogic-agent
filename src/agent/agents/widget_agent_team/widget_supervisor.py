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
        """Analyze the current state comprehensively, filtering large fields intelligently."""
        # Collect handoff messages from recent node executions
        handoff_messages = []
        
        # Try to get messages - state might be GraphState at runtime with messages field
        messages = getattr(state, 'messages', [])
        if messages:
            # Extract the last few tool messages as handoff messages
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            for msg in recent_messages:
                # Check if it's a tool message (from node executions)
                if hasattr(msg, 'type') and msg.type == 'tool':
                    handoff_messages.append({
                        "tool_call_id": getattr(msg, 'tool_call_id', 'unknown'),
                        "content": msg.content[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content),
                        "message_type": "tool"
                    })
                # Also include any other relevant message types for context
                elif hasattr(msg, 'type'):
                    handoff_messages.append({
                        "tool_call_id": getattr(msg, 'tool_call_id', 'none'),
                        "content": str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content),
                        "message_type": msg.type
                    })
        
        analysis = {
            "user_request": {
                "user_prompt": state.user_prompt,
                "task_instructions": state.task_instructions,
                "operation": state.operation,
                "widget_type": state.widget_type,
            },
            "task_progress": {
                "has_raw_data": state.raw_file_data is not None,
                "has_generated_code": state.generated_code is not None,
                "has_execution_result": state.code_execution_result is not None,
                "is_validated": state.data_validated,
                "has_errors": len(state.error_messages) > 0,
                "iteration_count": state.iteration_count,
                "current_status": state.task_status,
                # Database operation completion flags
                "db_create_completed": state.widget_creation_completed,
                "db_update_completed": state.widget_update_completed, 
                "db_delete_completed": state.widget_deletion_completed,
            },
            "handoff_messages": handoff_messages,
            "error_context": {
                "recent_errors": state.error_messages[-3:] if state.error_messages else [],
                "total_error_count": len(state.error_messages),
            },
            "data_context": {
                "has_file_schemas": len(state.file_schemas) > 0,
                "has_sample_data": len(state.file_sample_data) > 0,
                "file_count": len(state.file_ids),
                "widget_configured": bool(state.title and state.description),
                "dashboard_id": state.dashboard_id,
            },
        }
        
        # Only include generated_code and code_execution_result if they are None or empty
        if not state.generated_code:
            analysis["generated_code"] = "None - needs generation"
        if not state.code_execution_result:
            analysis["code_execution_result"] = "None - needs execution"
        elif isinstance(state.code_execution_result, dict) and "error" in state.code_execution_result:
            analysis["code_execution_result"] = f"Error: {state.code_execution_result['error'][:200]}..."
        
        return analysis

    def create_routing_prompt(self, state_analysis: Dict[str, Any]) -> str:
        """Create comprehensive routing prompt for LLM supervisor."""
        return f"""
You are an intelligent supervisor managing a widget data processing and database persistence pipeline. 


AVAILABLE NODES AND THEIR FUNCTIONS:
1. "data" - Unified data processing node that:
   - Fetches data from files if needed
   - Generates Python code for data analysis/transformation
   - Executes the code using E2B sandbox
   - Returns structured ChartConfigSchema results

2. "validate_data" - Data validation node that:
   - Uses LLM to validate execution results against user requirements
   - Provides confidence scoring (0-100%)
   - Gives detailed feedback for improvements if validation fails
   - Continues workflow to database operations if validation succeeds
   - Never go to this node if we have concluded with the database operations node.

3. "db_operations_node" - Database persistence node that:
   - Handles CREATE/UPDATE/DELETE operations for widgets
   - Uses create_widget, update_widget, delete_widget from dashboard.py
   - Persists validated data to the database
   - Marks task as completed after successful database operation
   - If the operation is DELETE, you go to the database operations node immediately.
   - After the database operations node, you should end the workflow.

4. "__end__" - Workflow termination:
   - Ends the workflow completely
   - Should only be used when task is truly complete or unrecoverable

ROUTING PHILOSOPHY:
Your job is to analyze the current state and user request to determine the next logical step. Consider:

- What the task instructions are (from task_instructions). You need to understand the instructions very well first and then make the best decision.
- What work has been completed (from task_progress and handoff_messages) 
- What errors occurred and need addressing (from error_context)
- A logical flow would be: data processing → validation → database operations → completion. However, you need to judge the situation and make the best decision. For example, if the widget_operation is DELETE, you should not go to the database operations node. Therefore, everytime, you need to think the most appropriate logical flow and next step..

IMPORTANT GUIDELINES:
- Base decisions on the user's intent and current progress, not hardcoded rules
- Use handoff messages to understand what just happened in the previous node
- Consider error recovery strategies when there are failures
- Route to database operations only after successful data validation
- Use retry logic thoughtfully (max 3-4 iterations before giving up)
- End the workflow only when the task is complete or clearly unrecoverable

COMPLETION SIGNALS - When to choose "end":
- If task_progress shows db_create_completed=true, db_update_completed=true, or db_delete_completed=true
- if you get a successfull message from the database operations node, you should end the workflow.
- Database operation completion means the ENTIRE TASK IS FINISHED.

CURRENT STATE ANALYSIS:
{json.dumps(state_analysis, indent=2)}

Analyze the situation and make an intelligent routing decision with clear reasoning about why this next step makes sense for achieving the user's goal.
"""

    def apply_business_rules(
        self, decision: SupervisorDecision, state: WidgetAgentState
    ) -> SupervisorDecision:
        """Apply minimal safety constraints to LLM decisions."""
        
        # Only apply critical safety constraints, let LLM handle logic
        
        # Safety: prevent infinite loops with absolute maximum
        if state.iteration_count >= 5 and decision.next_node in ["data", "validate_data"]:
            decision.next_node = "end"
            decision.reasoning = (
                f"Safety override: {decision.reasoning}. "
                "Maximum iterations (5) reached to prevent infinite loops."
            )

        return decision

    def __call__(self, state: WidgetAgentState) -> Command:
        """
        Main supervisor function that uses LLM to make all routing decisions based on state and user intent.
        """
        try:
            # Comprehensive state analysis
            state_analysis = self.analyze_state(state)

            # Create detailed prompt for LLM supervisor
            routing_prompt = self.create_routing_prompt(state_analysis)

            # Get structured decision from LLM
            decision = self.llm_with_structure.invoke(routing_prompt)

            # Apply minimal safety constraints only
            decision = self.apply_business_rules(decision, state)

            # Handle end condition
            if decision.next_node == "end":
                # Let the LLM determine final status based on what was accomplished
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
