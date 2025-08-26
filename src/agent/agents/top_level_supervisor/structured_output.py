"""Structured output models for the Top Level Supervisor.

This module defines Pydantic models to ensure structured output from the supervisor agent.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from agent.models import DelegatedTask


class SupervisorAnalysis(BaseModel):
    """Structured analysis output from the top level supervisor."""
    
    user_intent: str = Field(
        description="Clear description of what the user wants to accomplish"
    )
    data_assessment: str = Field(
        description="Summary of available data and its relevance to the request"
    )
    strategy: str = Field(
        description="Strategic approach for fulfilling the user's request"
    )
    

class TaskPlan(BaseModel):
    """Structured task planning output."""
    
    task_count: int = Field(
        description="Number of tasks that will be created"
    )
    reasoning: str = Field(
        description="Explanation of why these specific tasks were chosen"
    )
    expected_outcome: str = Field(
        description="What the user should expect when all tasks are completed"
    )


class SupervisorDecision(BaseModel):
    """Structured decision output from the supervisor agent."""
    
    action: str = Field(
        description="Next action the supervisor will take: 'analyze_data', 'create_tasks', 'monitor_tasks', or 'finalize'"
    )
    analysis: Optional[SupervisorAnalysis] = Field(
        default=None, description="Analysis results if action is 'analyze_data'"
    )
    task_plan: Optional[TaskPlan] = Field(
        default=None, description="Task planning if action is 'create_tasks'"
    )
    reasoning: str = Field(
        description="Reasoning behind this decision"
    )
    

class TaskCreationRequest(BaseModel):
    """Structured request for creating a specific widget task."""
    
    widget_type: str = Field(
        description="Type of widget to create (line, bar, pie, area, radial, kpi, table, text)"
    )
    operation: str = Field(
        description="Operation to perform (CREATE, UPDATE, DELETE)"
    )
    title: str = Field(
        description="Clear, descriptive title for the widget"
    )
    description: str = Field(
        description="Detailed description of what the widget will show"
    )
    file_ids: List[str] = Field(
        description="List of file IDs to use for this widget"
    )
    widget_id: Optional[str] = Field(
        default=None, description="Widget ID for UPDATE/DELETE operations"
    )
    reference_widget_id: Optional[str] = Field(
        default=None, description="Widget ID to reference for text blocks (e.g., chart widget being explained)"
    )
    task_instructions: str = Field(
        description="Detailed instructions for the widget_agent_team"
    )


class TaskCreationPlan(BaseModel):
    """Structured plan for creating multiple tasks."""
    
    tasks: List[TaskCreationRequest] = Field(
        description="List of tasks to be created"
    )
    strategy_summary: str = Field(
        description="Overall strategy explaining how these tasks work together"
    )
    duplicate_check: str = Field(
        description="Confirmation that no duplicate tasks are being created"
    )


class SupervisorResponse(BaseModel):
    """Final structured response from the top level supervisor."""
    
    status: str = Field(
        description="Current status: 'analyzing', 'planning', 'delegating', 'monitoring', 'completed', or 'failed'"
    )
    message: str = Field(
        description="Human-readable message about what was accomplished or is happening"
    )
    decision: Optional[SupervisorDecision] = Field(
        default=None, description="Decision details if still processing"
    )
    task_creation_plan: Optional[TaskCreationPlan] = Field(
        default=None, description="Task creation plan if delegating tasks"
    )
    tasks_summary: Optional[str] = Field(
        default=None, description="Summary of current delegated tasks status"
    )
    next_steps: Optional[str] = Field(
        default=None, description="What will happen next"
    )