"""Pydantic models for chart generation state and data structures."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CreateWidgetInput(BaseModel):
    """Input model for creating widgets."""

    dashboard_id: str = Field(description="Dashboard identifier")
    title: str = Field(description="Widget title")
    widget_type: str = Field(
        description="Widget type ('text', 'chart', 'kpi', 'table')"
    )
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


class UpdateWidgetInput(BaseModel):
    """Input model for updating widgets."""

    widget_id: str = Field(description="Widget identifier")
    title: Optional[str] = Field(default=None, description="Widget title")
    widget_type: Optional[str] = Field(default=None, description="Widget type")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Widget configuration"
    )
    data: Optional[Dict[str, Any]] = Field(default=None, description="Widget data")
    sql: Optional[str] = Field(default=None, description="SQL query")
    layout: Optional[Dict[str, Any]] = Field(
        default=None, description="React Grid Layout position"
    )
    order: Optional[int] = Field(default=None, description="Order for positioning")
    is_configured: Optional[bool] = Field(
        default=None, description="Configuration status"
    )
    cache_key: Optional[str] = Field(default=None, description="Cache key")


class BackendPayload(BaseModel):
    """Input payload from the frontend."""

    message: str = Field(description="User's natural language prompt")
    dashboard_id: str = Field(description="Dashboard to work with")
    context_widget_ids: Optional[List[str]] = Field(
        default=None, description="Existing widgets user is referencing"
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
        "update_task",
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
    widget_type: Literal["line", "bar", "pie", "area", "radial", "kpi", "table"]
    widget_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

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
    error_messages: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    current_step: Optional[str] = None
    widget_supervisor_reasoning: Optional[str] = None

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


class ChartConfigSchema(BaseModel):
    """Complete chart configuration schema."""

    chartType: Literal["line", "bar", "pie", "area", "radial", "kpi", "table"] = Field(
        description="Type of chart"
    )

    title: str = Field(description="Title of the chart")

    description: str = Field(description="Description of the chart")

    data: Dict[str, Any] = Field(description="Data for the chart")

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


# Chat Models for chat functionality


class ConversationMessage(BaseModel):
    """Individual message within a chat conversation."""

    role: Literal["user", "assistant", "system"] = Field(
        description="Role of the message sender"
    )
    message: str = Field(description="Content of the message")
    timestamp: str = Field(description="ISO timestamp when message was created")
    context_widget_ids: Optional[List[str]] = Field(
        default=None, description="Widget IDs referenced in this message"
    )
    target_widget_type: Optional[str] = Field(
        default=None, description="Target widget type for this message"
    )
    target_chart_sub_type: Optional[str] = Field(
        default=None, description="Target chart subtype for this message"
    )


class ChatMessageInput(BaseModel):
    """Input model for creating chat messages."""

    chat_id: str = Field(description="ID of the chat to append message to")
    role: Literal["user", "assistant", "system"] = Field(
        description="Role of the message sender"
    )
    message: str = Field(description="Content of the message")
    context_widget_ids: Optional[List[str]] = Field(
        default=None, description="Widget IDs referenced in this message"
    )
    target_widget_type: Optional[str] = Field(
        default=None, description="Target widget type for this message"
    )
    target_chart_sub_type: Optional[str] = Field(
        default=None, description="Target chart subtype for this message"
    )


class Chat(BaseModel):
    """Chat model representing a conversation in the database."""

    id: str = Field(description="Unique identifier for the chat")
    user_id: str = Field(description="ID of the user who owns this chat")
    dashboard_id: str = Field(description="ID of the dashboard this chat is associated with")
    title: Optional[str] = Field(default=None, description="Title of the chat")
    conversation: List[Dict[str, Any]] = Field(
        default_factory=list, description="Array of conversation messages"
    )
    created_at: str = Field(description="ISO timestamp when chat was created")
    updated_at: str = Field(description="ISO timestamp when chat was last updated")


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
    job_type: str = Field(description="Type of job (e.g., 'widget_creation', 'data_processing')")
    status: str = Field(default="pending", description="Initial job status")
    progress: int = Field(default=0, description="Job progress percentage (0-100)")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional job metadata"
    )


class UpdateJobInput(BaseModel):
    """Input model for updating jobs."""

    job_id: str = Field(description="Job identifier")
    status: Optional[str] = Field(default=None, description="Job status")
    progress: Optional[int] = Field(default=None, description="Job progress percentage (0-100)")
    error: Optional[str] = Field(default=None, description="Error message if job failed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Job result data")
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
    status: str = Field(description="Current job status (pending, processing, completed, failed)")
    progress: int = Field(description="Job progress percentage (0-100)")
    error: Optional[str] = Field(default=None, description="Error message if job failed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Job result data")
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
