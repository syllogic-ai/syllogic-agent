"""Widget supervisor node with intelligent routing based on complete state analysis."""

import json
from datetime import datetime
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command

from agent.models import SupervisorDecision, WidgetAgentState


class WidgetSupervisor:
    """Intelligent supervisor that analyzes complete state and routes to appropriate nodes."""

    def __init__(self, llm_model: str = "openai:gpt-4o-mini"):
        """Initialize the supervisor with LLM."""
        self.llm = ChatOpenAI(model="gpt-4o-mini")
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
                    tool_call_id = getattr(msg, 'tool_call_id', 'unknown')
                    
                    # Determine source node from tool_call_id patterns
                    source_node = self._identify_source_node(tool_call_id)
                    
                    handoff_messages.append({
                        "tool_call_id": tool_call_id,
                        "source_node": source_node,
                        "content": msg.content[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content),
                        "message_type": "tool"
                    })
                # Also include any other relevant message types for context
                elif hasattr(msg, 'type'):
                    handoff_messages.append({
                        "tool_call_id": getattr(msg, 'tool_call_id', 'none'),
                        "source_node": "unknown",
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
            "previous_node_info": self._get_previous_node_info(handoff_messages),
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

    def _identify_source_node(self, tool_call_id: str) -> str:
        """
        Identify the source node from tool_call_id patterns.
        Each node uses specific tool call ID patterns.
        """
        if not tool_call_id or tool_call_id == 'unknown':
            return "unknown"
            
        # Database operations node patterns
        if any(pattern in tool_call_id for pattern in ['db_create_complete', 'db_update_complete', 'db_delete_complete']):
            return "db_operations_node"
            
        # Validation node patterns
        if any(pattern in tool_call_id for pattern in ['validation_complete', 'validation_failed']):
            return "validate_data"
            
        # Data processing node patterns (from worker_nodes.py tools)
        if any(pattern in tool_call_id for pattern in ['fetch_data', 'generate_python_code', 'e2b_sandbox']):
            return "data"
            
        # Check for other common patterns
        if 'fetch' in tool_call_id or 'data' in tool_call_id:
            return "data"
        elif 'validate' in tool_call_id or 'validation' in tool_call_id:
            return "validate_data"
        elif 'db' in tool_call_id or 'database' in tool_call_id or 'create' in tool_call_id or 'update' in tool_call_id or 'delete' in tool_call_id:
            return "db_operations_node"
            
        return "unknown"

    def _get_previous_node_info(self, handoff_messages: list) -> Dict[str, str]:
        """
        Determine the most recent previous node and its outcome from handoff messages.
        """
        if not handoff_messages:
            return {"previous_node": "none", "outcome": "no_previous_execution"}
            
        # Get the most recent tool message (last executed node)
        recent_tool_messages = [msg for msg in handoff_messages if msg.get("message_type") == "tool"]
        if not recent_tool_messages:
            return {"previous_node": "none", "outcome": "no_tool_messages"}
            
        last_message = recent_tool_messages[-1]
        previous_node = last_message.get("source_node", "unknown")
        
        # Determine outcome from message content
        content = last_message.get("content", "").lower()
        if "complete" in content or "successfully" in content:
            outcome = "success"
        elif "error" in content or "failed" in content:
            outcome = "error"
        else:
            outcome = "unknown"
            
        return {
            "previous_node": previous_node,
            "outcome": outcome,
            "last_message_summary": last_message.get("content", "")[:100] + "..." if len(last_message.get("content", "")) > 100 else last_message.get("content", "")
        }

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
- Which node just executed and what was the outcome (from previous_node_info)
- What errors occurred and need addressing (from error_context)
- A logical flow would be: data processing → validation → database operations → completion. However, you need to judge the situation and make the best decision. For example, if the operation is DELETE, you should go directly to the database operations node without data processing or validation. Therefore, everytime, you need to think the most appropriate logical flow and next step based on the operation type.

PREVIOUS NODE CONTEXT:
Pay special attention to the "previous_node_info" which tells you:
- Which node just executed (previous_node)
- Whether it succeeded or failed (outcome)
- A summary of what happened (last_message_summary)
This helps you understand the immediate context for your routing decision.

IMPORTANT GUIDELINES:
- Base decisions on the user's intent and current progress, not hardcoded rules
- Use handoff messages to understand what just happened in the previous node
- Consider error recovery strategies when there are failures
- Route to database operations only after successful data validation
- Use retry logic thoughtfully (max 3-4 iterations before giving up)
- End the workflow only when the task is complete or clearly unrecoverable

COMPLETION SIGNALS - When to choose "end":
- If task_progress shows db_create_completed=true, db_update_completed=true, or db_delete_completed=true, the workflow MUST end immediately
- If you get a successful message from the database operations node, you should end the workflow
- Database operation completion means the ENTIRE TASK IS FINISHED - no further processing needed
- CRITICAL: If any db_*_completed flag is true, you MUST route to "end" regardless of other state

CURRENT STATE ANALYSIS:
{json.dumps(state_analysis, indent=2)}

Analyze the situation and make an intelligent routing decision with clear reasoning about why this next step makes sense for achieving the user's goal.
"""

    def apply_business_rules(
        self, decision: SupervisorDecision, state: WidgetAgentState
    ) -> SupervisorDecision:
        """Apply minimal safety constraints to LLM decisions."""
        
        # Critical: Force end if any database operation is completed
        if (state.widget_creation_completed or 
            state.widget_update_completed or 
            state.widget_deletion_completed):
            decision.next_node = "end"
            decision.reasoning = (
                f"Business rule override: Database operation completed "
                f"(create={state.widget_creation_completed}, update={state.widget_update_completed}, "
                f"delete={state.widget_deletion_completed}). Workflow must end."
            )
            return decision
        
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
            # Check for immediate completion before expensive LLM calls
            if (state.widget_creation_completed or 
                state.widget_update_completed or 
                state.widget_deletion_completed):
                # Database operation completed - create end decision directly
                decision = SupervisorDecision(
                    next_node="end",
                    reasoning=(
                        f"Database operation completed successfully "
                        f"(create={state.widget_creation_completed}, update={state.widget_update_completed}, "
                        f"delete={state.widget_deletion_completed}). Task is finished."
                    )
                )
            else:
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
                # Determine final status based on what was accomplished
                if (state.widget_creation_completed or 
                    state.widget_update_completed or 
                    state.widget_deletion_completed):
                    final_status = "completed"
                elif state.data_validated:
                    final_status = "completed"
                elif len(state.error_messages) > 0:
                    final_status = "failed"
                else:
                    final_status = "failed"  # Default fallback
                
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
