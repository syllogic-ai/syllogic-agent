"""Main entry point for the Top Level Supervisor Agent Team.

This module provides the TopLevelSupervisorRunner class for orchestrating
multi-agent workflows and delegating tasks to specialized agent teams.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from agent.models import TopLevelSupervisorState, BackendPayload
from .top_level_supervisor import top_level_supervisor

logger = logging.getLogger(__name__)


class TopLevelSupervisorRunner:
    """Runner class for the Top Level Supervisor agent team.
    
    This class manages the lifecycle of supervisor operations, from initial
    request processing through task delegation and completion tracking.
    """
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        """Initialize the supervisor runner.
        
        Args:
            model_name: The LLM model to use for the supervisor
        """
        self.model_name = model_name
        logger.info(f"Initialized TopLevelSupervisorRunner with model: {model_name}")
    
    def create_initial_state(self, payload: BackendPayload) -> TopLevelSupervisorState:
        """Create initial supervisor state from backend payload.
        
        Args:
            payload: Backend payload containing user request and context
            
        Returns:
            Initialized TopLevelSupervisorState
        """
        try:
            state = TopLevelSupervisorState(
                user_prompt=payload.message,
                user_id=payload.user_id,
                dashboard_id=payload.dashboard_id,
                chat_id=payload.chat_id,
                request_id=payload.request_id,
                file_ids=payload.file_ids,
                context_widget_ids=payload.context_widget_ids,
                supervisor_status="analyzing",
                created_at=datetime.now()
            )
            
            logger.info(f"Created initial supervisor state for request: {payload.request_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error creating initial state: {e}")
            raise
    
    def run_supervisor_cycle(self, state: TopLevelSupervisorState) -> TopLevelSupervisorState:
        """Run a single supervisor analysis and delegation cycle.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Updated supervisor state after processing
        """
        try:
            logger.info(f"Running supervisor cycle for request: {state.request_id}")
            
            # Update supervisor status
            state.supervisor_status = "analyzing"
            
            # Run the supervisor agent
            updates = top_level_supervisor(state)
            
            # Apply updates to state and store structured response
            structured_response = updates.get("structured_response")
            for key, value in updates.items():
                if key != "structured_response" and hasattr(state, key):
                    setattr(state, key, value)
            
            # Store the structured response for later use (as a simple attribute)
            # Note: Since TopLevelSupervisorState is a Pydantic model, we store as a dict
            if structured_response:
                state._latest_structured_response = structured_response
            
            logger.info(f"Completed supervisor cycle. Status: {state.supervisor_status}")
            return state
            
        except Exception as e:
            logger.error(f"Error in supervisor cycle: {e}")
            state.error_messages.append(f"Supervisor cycle error: {str(e)}")
            state.supervisor_status = "failed"
            state.updated_at = datetime.now()
            return state
    
    def monitor_and_coordinate(
        self, 
        state: TopLevelSupervisorState,
        max_iterations: int = 10
    ) -> TopLevelSupervisorState:
        """Monitor delegated tasks and coordinate until completion.
        
        Args:
            state: Current supervisor state
            max_iterations: Maximum coordination iterations
            
        Returns:
            Final supervisor state
        """
        try:
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Coordination iteration {iteration} for request: {state.request_id}")
                
                # Check if we're done
                if state.supervisor_status in ["completed", "failed"]:
                    break
                
                # If we have tasks but none are completed yet, continue monitoring
                if state.delegated_tasks and not state.all_tasks_completed:
                    state.supervisor_status = "monitoring"
                    
                    # In a real implementation, this would check with the actual
                    # widget_agent_team for task completion status
                    # For now, we'll simulate task completion tracking
                    logger.info(f"Monitoring {len(state.delegated_tasks)} delegated tasks")
                    
                    # Update the timestamp
                    state.updated_at = datetime.now()
                    
                    # Break for now - in production this would wait and check again
                    break
                
                # If no tasks have been created yet, we might need another supervisor cycle
                if not state.delegated_tasks and state.supervisor_status == "analyzing":
                    logger.info("No tasks delegated yet, running another supervisor cycle")
                    state = self.run_supervisor_cycle(state)
                    continue
                    
                # If all tasks are completed, finalize
                if state.all_tasks_completed:
                    state.supervisor_status = "completed"
                    if not state.final_response:
                        state.final_response = "All requested tasks have been completed successfully."
                    state.updated_at = datetime.now()
                    break
            
            if iteration >= max_iterations:
                logger.warning(f"Reached maximum iterations ({max_iterations}) for request: {state.request_id}")
                state.error_messages.append("Reached maximum coordination iterations")
                state.supervisor_status = "failed"
                state.updated_at = datetime.now()
            
            return state
            
        except Exception as e:
            logger.error(f"Error in monitor and coordinate: {e}")
            state.error_messages.append(f"Coordination error: {str(e)}")
            state.supervisor_status = "failed"
            state.updated_at = datetime.now()
            return state
    
    def execute_request(
        self, 
        payload: BackendPayload,
        max_coordination_iterations: int = 10
    ) -> Dict[str, Any]:
        """Execute a complete request from start to finish.
        
        Args:
            payload: Backend payload containing user request
            max_coordination_iterations: Maximum coordination cycles
            
        Returns:
            Final result with supervisor state and execution summary
        """
        try:
            logger.info(f"Executing request: {payload.request_id}")
            
            # Create initial state
            state = self.create_initial_state(payload)
            
            # Run initial supervisor analysis and delegation
            cycle_result = self.run_supervisor_cycle(state)
            
            # Monitor and coordinate until completion
            state = self.monitor_and_coordinate(state, max_coordination_iterations)
            
            # Get the latest structured response from the supervisor cycle  
            latest_structured_response = getattr(state, '_latest_structured_response', None)
            
            # Prepare result with structured output
            result = {
                "request_id": state.request_id,
                "status": state.supervisor_status,
                "final_response": state.final_response or "Request processed",
                "delegated_tasks": [
                    {
                        "task_id": task.task_id,
                        "target_agent": task.target_agent,
                        "widget_type": task.widget_type,
                        "operation": task.operation,
                        "status": task.task_status,
                        "instructions": task.task_instructions,
                        "result": task.result
                    }
                    for task in state.delegated_tasks
                ],
                "available_data_summary": state.available_data_summary,
                "error_messages": state.error_messages,
                "created_at": state.created_at.isoformat(),
                "updated_at": state.updated_at.isoformat() if state.updated_at else None,
                "structured_response": latest_structured_response.model_dump() if latest_structured_response else None
            }
            
            logger.info(f"Completed request execution: {payload.request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing request: {e}")
            return {
                "request_id": payload.request_id if hasattr(payload, 'request_id') else "unknown",
                "status": "failed",
                "final_response": "Request execution failed",
                "error_messages": [str(e)],
                "delegated_tasks": [],
                "available_data_summary": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
    
    def get_task_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a request and its tasks.
        
        Args:
            request_id: The request identifier
            
        Returns:
            Status information or None if not found
            
        Note:
            In a production system, this would query a persistent store
            to retrieve the current state of the request.
        """
        # This is a placeholder - in production you'd query a database
        # or state store to get the current status
        logger.info(f"Status check requested for: {request_id}")
        return {
            "request_id": request_id,
            "status": "This would be retrieved from persistent storage",
            "message": "Status checking requires integration with state persistence"
        }