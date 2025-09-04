"""Pydantic models for chart generation state and data structures."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Annotated, Sequence

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CreateWidgetInput(BaseModel):
    """Input model for creating widgets."""

    dashboard_id: str = Field(description="Dashboard identifier")
    title: str = Field(description="Widget title")
    widget_type: str = Field(
        description="Widget type ('text', 'chart', 'kpi', 'table')"
    )
    description: Optional[str] = Field(default=None, description="Widget description")
    config: Dict[str, Any] = Field(description="Widget configuration")
    widget_id: Optional[str] = Field(default=None, description="Specific widget ID to use (if not provided, generates new UUID)")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Widget data")
    order: Optional[int] = Field(default=None, description="Order for positioning")
    summary: Optional[str] = Field(
        default=None, description="Brief VLLM-friendly widget summary"
    )
    is_configured: Optional[bool] = Field(
        default=None, description="Configuration status"
    )


class UpdateWidgetInput(BaseModel):
    """Input model for updating widgets."""

    widget_id: str = Field(description="Widget identifier")
    title: Optional[str] = Field(default=None, description="Widget title")
    widget_type: Optional[str] = Field(default=None, description="Widget type")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Widget configuration"
    )
    data: Optional[Dict[str, Any]] = Field(default=None, description="Widget data")
    order: Optional[int] = Field(default=None, description="Order for positioning")
    is_configured: Optional[bool] = Field(
        default=None, description="Configuration status"
    )
    summary: Optional[str] = Field(
        default=None, description="Brief VLLM-friendly widget summary"
    )


class BackendPayload(BaseModel):
    """Input payload from the frontend."""

    message: str = Field(description="User's natural language prompt")
    dashboard_id: str = Field(description="Dashboard to work with")
    context_widget_ids: Optional[List[str]] = Field(
        default=None, description="Existing widgets user is referencing"
    )
    file_ids: List[str] = Field(
        default_factory=list, description="File IDs available for widget creation"
    )
    chat_id: str = Field(description="For conversation continuity")
    request_id: str = Field(description="For logging/tracking")
    user_id: str = Field(description="For personalization")
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Full chat history for context"
    )


class ColumnInfo(BaseModel):
    """Information about a data column."""

    name: str
    type: str
    null_count: int
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list)


class FileSchema(BaseModel):
    """Schema information for a data file."""

    file_id: str
    columns: List[ColumnInfo]
    total_rows: int
    total_columns: int


class FileSampleData(BaseModel):
    """Sample data from a file."""

    file_id: str
    headers: List[str]
    rows: List[List[Any]]
    total_rows_in_file: int
    sample_rows_returned: int


class SupervisorDecision(BaseModel):
    """Structured output for widget_supervisor routing decisions"""

    next_node: Literal[
        "data",
        "validate_data", 
        "text_block_node",
        "end",
    ]
    reasoning: str
    retry_with_modification: bool = False
    modification_hint: Optional[str] = None


class WidgetAgentState(BaseModel):
    """Complete state for the widget agent workflow using Pydantic"""

    # Original WidgetTask fields
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending"
    )
    task_instructions: str
    user_prompt: str

    # Widget configuration
    operation: Literal["CREATE", "UPDATE", "DELETE"]
    widget_type: Literal["line", "bar", "pie", "area", "radial", "kpi", "table", "text"]
    widget_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reference_widget_id: List[str] = Field(
        default_factory=list,
        description="Array of widget IDs that this task depends on. When dependent tasks complete and create widgets, their widget_ids are added to this array."
    )
    dashboard_id: str = Field(description="Dashboard identifier for the widget")
    chat_id: Optional[str] = Field(default=None, description="Chat ID if created from chat")

    # File context
    file_ids: List[str] = Field(default_factory=list)
    file_sample_data: List[FileSampleData] = Field(default_factory=list)
    file_schemas: List[FileSchema] = Field(default_factory=list)

    # Widget metadata
    title: str
    description: str

    # Task result fields
    data: Optional[Dict[str, Any]] = None
    widget_metadata: Optional[Dict[str, Any]] = None
    data_validated: bool = False

    # Agent workflow fields
    raw_file_data: Optional[Dict[str, Any]] = None
    generated_code: Optional[str] = None
    code_execution_result: Optional[Any] = None
    widget_config: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Unified widget configuration ready for validation and database persistence. "
                   "For text blocks: {'content': html_string}. For charts: ChartConfigSchema result."
    )
    error_messages: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    current_step: Optional[str] = None
    widget_supervisor_reasoning: Optional[str] = None
    
    # Database operation completion flags
    widget_creation_completed: bool = False
    widget_update_completed: bool = False
    widget_deletion_completed: bool = False
    
    # Database operation object for top-level supervisor
    database_operation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Database operation object to be executed by top-level supervisor"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


# Chart Config Schema


class ChartItem(BaseModel):
    """Individual item configuration within chartConfig."""

    label: str = Field(description="Display label for the chart item")
    color: str = Field(
        description="Color value for the chart item (e.g., 'var(--chart-1)' or hex)"
    )


class XAxisConfig(BaseModel):
    """Configuration for the chart's X-axis."""

    dataKey: str = Field(description="The data key to use for the X-axis")


class TextBlockContentSchema(BaseModel):
    """Content for a text block."""

    content: str = Field(description="HTML content of the text block")


class ChartConfigSchema(BaseModel):
    """Complete chart configuration schema."""

    chartType: Literal["line", "bar", "pie", "area", "radial", "kpi", "table"] = Field(
        description="Type of chart"
    )

    title: str = Field(description="Title of the chart")

    description: str = Field(description="Description of the chart")

    data: List[Dict[str, Any]] = Field(description="Data for the chart")

    chartConfig: Dict[str, ChartItem] = Field(
        description="Dictionary of chart items, where keys are item names and values contain label and color"
    )
    xAxisConfig: XAxisConfig = Field(description="X-axis configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "category": "2023",  # example data and schema. actual data will be different.
                        "revenue": 100000,  # example data and schema. actual data will be different.
                        "cost": 50000,  # example data and schema. actual data will be different.
                        "profit": 50000,  # example data and schema. actual data will be different.
                    },
                    {
                        "category": "2024",  # example data and schema. actual data will be different.
                        "revenue": 150000,  # example data and schema. actual data will be different.
                        "cost": 75000,  # example data and schema. actual data will be different.
                        "profit": 75000,  # example data and schema. actual data will be different.
                    },
                ],
                "chartConfig": {
                    "revenue": {"label": "Revenue", "color": "var(--chart-1)"},
                    "cost": {"label": "Cost", "color": "var(--chart-2)"},
                    "profit": {"label": "Profit", "color": "var(--chart-3)"},
                },
                "xAxisConfig": {"dataKey": "category"},
            }
        }


# Chat and Message Models


class Message(BaseModel):
    """Message model representing individual chat messages in the database."""

    id: str = Field(description="Unique identifier for the message")
    chat_id: str = Field(description="ID of the chat this message belongs to")
    role: Literal["user", "ai", "system"] = Field(
        description="Role of the message sender"
    )
    content: Optional[str] = Field(default=None, description="Content of the message")
    message_type: Optional[str] = Field(
        default=None, description="Type of message ('chat', 'task-list', 'tool-usage')"
    )
    task_group_id: Optional[str] = Field(
        default=None, description="Links message to a group of related tasks"
    )
    created_at: str = Field(description="ISO timestamp when message was created")
    updated_at: str = Field(description="ISO timestamp when message was last updated")


class CreateMessageInput(BaseModel):
    """Input model for creating messages."""

    chat_id: str = Field(description="ID of the chat to append message to")
    role: Literal["user", "ai", "system"] = Field(
        description="Role of the message sender"
    )
    content: Optional[str] = Field(default=None, description="Content of the message")
    message_type: Optional[str] = Field(
        default="chat", description="Type of message ('chat', 'task-list', 'tool-usage')"
    )
    task_group_id: Optional[str] = Field(
        default=None, description="Links message to a group of related tasks"
    )


class Chat(BaseModel):
    """Chat model representing a conversation in the database."""

    id: str = Field(description="Unique identifier for the chat")
    user_id: str = Field(description="ID of the user who owns this chat")
    dashboard_id: str = Field(
        description="ID of the dashboard this chat is associated with"
    )
    title: Optional[str] = Field(default="Dashboard Chat", description="Title of the chat")
    last_message_at: Optional[str] = Field(
        default=None, description="ISO timestamp of last message"
    )
    message_count: int = Field(default=0, description="Number of messages in chat")
    created_at: str = Field(description="ISO timestamp when chat was created")
    updated_at: str = Field(description="ISO timestamp when chat was last updated")


# Task Models


class Task(BaseModel):
    """Task model representing job tasks and progress in the database."""

    id: str = Field(description="Unique identifier for the task")
    chat_id: str = Field(description="ID of the chat this task belongs to")
    dashboard_id: str = Field(description="Dashboard identifier for the task")
    task_group_id: str = Field(
        description="Group ID linking multiple tasks from same AI response"
    )
    title: str = Field(description="Task title (e.g., 'Fetch data', 'Generate code')")
    description: Optional[str] = Field(
        default=None, description="Optional detailed description"
    )
    status: Literal["pending", "in-progress", "completed", "failed"] = Field(
        default="pending", description="Current task status"
    )
    order: int = Field(description="Display order within the task group")
    started_at: Optional[str] = Field(
        default=None, description="ISO timestamp when task started"
    )
    completed_at: Optional[str] = Field(
        default=None, description="ISO timestamp when task completed"
    )
    created_at: str = Field(description="ISO timestamp when task was created")
    updated_at: str = Field(description="ISO timestamp when task was last updated")


class CreateTaskInput(BaseModel):
    """Input model for creating tasks."""

    chat_id: str = Field(description="ID of the chat this task belongs to")
    dashboard_id: str = Field(description="Dashboard identifier for the task")
    task_group_id: str = Field(
        description="Group ID linking multiple tasks from same AI response"
    )
    title: str = Field(description="Task title")
    description: Optional[str] = Field(default=None, description="Task description")
    status: Literal["pending", "in-progress", "completed", "failed"] = Field(
        default="pending", description="Initial task status"
    )
    order: int = Field(description="Display order within the task group")


class UpdateTaskInput(BaseModel):
    """Input model for updating tasks."""

    task_id: str = Field(description="Task identifier")
    status: Optional[Literal["pending", "in-progress", "completed", "failed"]] = Field(
        default=None, description="Task status"
    )
    started_at: Optional[str] = Field(
        default=None, description="ISO timestamp when task started"
    )
    completed_at: Optional[str] = Field(
        default=None, description="ISO timestamp when task completed"
    )


class Widget(BaseModel):
    """Widget model representing a dashboard widget in the database."""

    id: str = Field(description="Unique identifier for the widget")
    dashboard_id: str = Field(description="Dashboard identifier")
    title: str = Field(description="Widget title")
    type: str = Field(description="Widget type ('text', 'chart', 'kpi', 'table')")
    config: Dict[str, Any] = Field(description="Widget configuration")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Widget data")
    sql: Optional[str] = Field(default=None, description="SQL query")
    layout: Optional[Dict[str, Any]] = Field(
        default=None, description="React Grid Layout position"
    )
    chat_id: Optional[str] = Field(
        default=None, description="Chat ID if created from chat"
    )
    order: Optional[int] = Field(default=None, description="Order for positioning")
    is_configured: Optional[bool] = Field(
        default=None, description="Configuration status"
    )
    cache_key: Optional[str] = Field(default=None, description="Cache key")
    summary: Optional[str] = Field(
        default=None, description="Brief VLLM-friendly widget summary for accessibility"
    )
    created_at: Optional[str] = Field(
        default=None, description="ISO timestamp when widget was created"
    )
    updated_at: Optional[str] = Field(
        default=None, description="ISO timestamp when widget was last updated"
    )


# Job Models for job management functionality


class CreateJobInput(BaseModel):
    """Input model for creating jobs."""

    job_id: str = Field(description="Unique identifier for the job")
    user_id: str = Field(description="ID of the user who owns this job")
    dashboard_id: Optional[str] = Field(
        default=None, description="Dashboard ID if job is related to a dashboard"
    )
    job_type: str = Field(
        description="Type of job (e.g., 'widget_creation', 'data_processing')"
    )
    status: str = Field(default="pending", description="Initial job status")
    progress: int = Field(default=0, description="Job progress percentage (0-100)")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional job metadata"
    )


class UpdateJobInput(BaseModel):
    """Input model for updating jobs."""

    job_id: str = Field(description="Job identifier")
    status: Optional[str] = Field(default=None, description="Job status")
    progress: Optional[int] = Field(
        default=None, description="Job progress percentage (0-100)"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if job failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Job result data"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional job metadata"
    )


class Job(BaseModel):
    """Job model representing a background job in the database."""

    id: str = Field(description="Unique identifier for the job")
    user_id: str = Field(description="ID of the user who owns this job")
    dashboard_id: Optional[str] = Field(
        default=None, description="Dashboard ID if job is related to a dashboard"
    )
    job_type: str = Field(description="Type of job")
    status: str = Field(
        description="Current job status (pending, processing, completed, failed)"
    )
    progress: int = Field(description="Job progress percentage (0-100)")
    error: Optional[str] = Field(
        default=None, description="Error message if job failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Job result data"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional job metadata"
    )
    created_at: str = Field(description="ISO timestamp when job was created")
    updated_at: str = Field(description="ISO timestamp when job was last updated")
    started_at: Optional[str] = Field(
        default=None, description="ISO timestamp when job processing started"
    )
    completed_at: Optional[str] = Field(
        default=None, description="ISO timestamp when job was completed"
    )


# Top Level Supervisor Models


class DelegatedTask(BaseModel):
    """A task that has been delegated to a specialized agent team."""
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_agent: Literal["widget_agent_team"] = Field(
        description="Which agent team should handle this task"
    )
    task_title: str = Field(description="Title of the task")
    task_instructions: str = Field(description="Detailed instructions for the task")
    task_status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending"
    )
    
    # Widget-specific task data
    widget_type: Literal["line", "bar", "pie", "area", "radial", "kpi", "table", "text"] = Field(
        description="Type of widget to create/update/delete"
    )
    operation: Literal["CREATE", "UPDATE", "DELETE"] = Field(
        description="Widget operation to perform"
    )
    file_ids: List[str] = Field(
        default_factory=list, description="File IDs to use for the widget"
    )
    widget_id: Optional[str] = Field(
        default=None, description="Widget ID for UPDATE/DELETE operations or context reference"
    )
    reference_widget_id: List[str] = Field(
        default_factory=list,
        description="Array of widget IDs that this task depends on. When dependent tasks complete and create widgets, their widget_ids are added to this array."
    )
    
    # Additional widget task fields (these contain all needed data for widget_agent_team)
    title: str = Field(description="Widget title")
    description: str = Field(description="Widget description")
    user_prompt: str = Field(description="Original user request")
    dashboard_id: str = Field(description="Dashboard ID where widget will be created")
    chat_id: str = Field(description="Chat ID for context")
    
    # Task database integration
    db_task_id: Optional[str] = Field(default=None, description="Associated database task ID")
    task_group_id: Optional[str] = Field(default=None, description="Task group ID for database")
    
    # Task results
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result data - database operation for completed tasks")
    error_message: Optional[str] = Field(default=None, description="Error if task failed")
    database_operation: Optional[Dict[str, Any]] = Field(default=None, description="Database operation to be executed by top-level supervisor")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskDependency(BaseModel):
    """Track dependencies between tasks."""
    task_id: str = Field(description="The task that has dependencies")
    dependent_on: List[str] = Field(default_factory=list, description="Task IDs this task depends on")  
    reference_widget_ids: List[str] = Field(default_factory=list, description="Widget IDs from completed dependent tasks")


class TopLevelSupervisorState(BaseModel):
    """State for the top-level supervisor that orchestrates all agent teams."""
    
    # Required LangGraph agent fields
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list, description="Message history for the conversation"
    )
    remaining_steps: int = Field(default=10, description="Remaining steps for task execution")
    
    # MINIMUM REQUIRED INPUT - Only these two fields are required
    user_prompt: str = Field(description="Original user request")
    dashboard_id: str = Field(description="Dashboard context for the request")
    
    # Optional request information - auto-generated if not provided
    user_id: str = Field(default_factory=lambda: f"user_{uuid.uuid4().hex[:8]}", description="ID of the user making the request")
    chat_id: str = Field(default_factory=lambda: f"chat_{uuid.uuid4().hex[:8]}", description="Chat ID for conversation continuity")
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}", description="Unique identifier for this request")
    
    # Optional context from backend payload
    file_ids: List[str] = Field(default_factory=list, description="File IDs from backend payload (optional - discovered if not provided)")
    context_widget_ids: Optional[List[str]] = Field(
        default=None, description="Existing widgets user is referencing (optional)"
    )
    
    # Available data context
    available_files: List[str] = Field(default_factory=list, description="List of available file IDs")
    file_schemas: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed file schema information")
    available_data_summary: Optional[str] = Field(
        default=None, description="Summary of available data for analysis"
    )
    
    # Task management
    delegated_tasks: List[DelegatedTask] = Field(
        default_factory=list, description="Tasks delegated to specialized agents"
    )
    completed_widget_ids: Dict[str, str] = Field(
        default_factory=dict, description="Mapping of task_id to actual widget_id for completed tasks"
    )
    task_dependencies: List[TaskDependency] = Field(
        default_factory=list, description="Track which tasks depend on which other tasks and their resolved widget IDs"
    )
    current_reasoning: Optional[str] = Field(
        default=None, description="Current reasoning and analysis"
    )
    
    # Status tracking
    supervisor_status: Literal["analyzing", "delegating", "monitoring", "completed", "failed"] = Field(
        default="analyzing"
    )
    all_tasks_completed: bool = Field(default=False)
    
    # Results
    final_response: Optional[str] = Field(
        default=None, description="Final response to the user"
    )
    error_messages: List[str] = Field(default_factory=list)
    
    # Error handling and retry tracking
    tool_failure_counts: Dict[str, int] = Field(default_factory=dict, description="Count of failures per tool")
    max_tool_retries: int = Field(default=3, description="Maximum retries per tool before giving up")
    last_failed_tool: Optional[str] = Field(default=None, description="Name of the last tool that failed")
    
    # Database operations queue - stores objects to be written to database
    pending_database_operations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Queue of database operations to be executed after all tasks complete"
    )
    
    # Database operations control flags
    auto_execute_database_ops: bool = Field(default=False, description="Flag to trigger automatic database operations execution")
    database_operations_executed: bool = Field(default=False, description="Flag tracking if database operations have been executed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
