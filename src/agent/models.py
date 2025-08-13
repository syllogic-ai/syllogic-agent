"""Pydantic models for chart generation state and data structures."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Database Models (matching Drizzle schema)


class User(BaseModel):
    """User model matching the database schema."""

    id: str = Field(description="Clerk user ID")
    email: str
    created_at: Optional[datetime] = None


class File(BaseModel):
    """File model matching the database schema."""

    id: str = Field(description="File identifier")
    user_id: str = Field(description="User who owns the file")
    dashboard_id: Optional[str] = Field(
        default=None, description="Associated dashboard ID"
    )
    file_type: str = Field(description="'original' | 'cleaned' | 'meta'")
    original_filename: str = Field(description="User-visible filename")
    sanitized_filename: Optional[str] = Field(
        default=None, description="UUID-based filename for storage"
    )
    storage_path: str = Field(description="Path in storage")
    mime_type: Optional[str] = Field(
        default=None, description="MIME type for proper display"
    )
    size: Optional[int] = Field(default=None, description="File size in bytes")
    status: str = Field(
        default="ready", description="'processing' | 'ready' | 'failed'"
    )
    created_at: Optional[datetime] = None


class Dashboard(BaseModel):
    """Dashboard model matching the database schema."""

    id: str = Field(description="Dashboard identifier")
    user_id: str = Field(description="User who owns the dashboard")
    name: str = Field(default="New Dashboard", description="Dashboard name")
    description: Optional[str] = Field(
        default=None, description="Dashboard description"
    )
    icon: str = Field(default="document-text", description="Dashboard icon")
    setup_completed: bool = Field(default=False, description="Setup state tracking")
    is_public: bool = Field(default=False, description="Public sharing")
    active_theme_id: Optional[str] = Field(default=None, description="Active theme ID")
    theme_mode: str = Field(default="light", description="'light' | 'dark' | 'system'")
    width: str = Field(default="full", description="'full' | 'constrained'")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Widget(BaseModel):
    """Widget model matching the database schema."""

    id: str = Field(description="Widget identifier")
    dashboard_id: str = Field(description="Associated dashboard ID")
    title: str = Field(description="Widget title")
    type: str = Field(description="'text' | 'chart' | 'kpi' | 'table'")
    config: Dict[str, Any] = Field(description="Widget configuration")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Widget data")
    sql: Optional[str] = Field(default=None, description="SQL query for the dashboard")
    layout: Optional[Dict[str, Any]] = Field(
        default=None, description="React Grid Layout position"
    )
    order: Optional[int] = Field(default=None, description="Order-based positioning")
    chat_id: Optional[str] = Field(
        default=None, description="Chat ID if created from chat"
    )
    is_configured: bool = Field(
        default=False, description="Widget configuration status"
    )
    cache_key: Optional[str] = Field(default=None, description="Cache key")
    last_data_fetch: Optional[datetime] = Field(
        default=None, description="Last data fetch timestamp"
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConversationMessage(BaseModel):
    """Individual message within a chat conversation."""

    role: str = Field(description="Message role")
    message: str = Field(description="Message content")
    timestamp: str = Field(description="Message timestamp")
    context_widget_ids: Optional[List[str]] = Field(
        default=None, description="Widgets included as context"
    )
    target_widget_type: Optional[str] = Field(
        default=None, description="'chart' | 'table' | 'kpi'"
    )
    target_chart_sub_type: Optional[str] = Field(
        default=None, description="'line' | 'area' | 'bar' | 'horizontal-bar' | 'pie'"
    )


class Chat(BaseModel):
    """Chat model matching the database schema."""

    id: str = Field(description="Chat identifier")
    user_id: str = Field(description="User who owns the chat")
    dashboard_id: str = Field(description="Associated dashboard ID")
    title: str = Field(default="Dashboard Chat", description="Chat title")
    conversation: List[ConversationMessage] = Field(description="Conversation messages")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Job(BaseModel):
    """Job model matching the database schema."""

    id: str = Field(description="UUID job ID")
    user_id: str = Field(description="User ID (stored as text)")
    dashboard_id: str = Field(description="Dashboard ID (stored as text)")
    status: str = Field(
        default="pending",
        description="'pending' | 'processing' | 'completed' | 'failed'",
    )
    progress: int = Field(default=0, description="0-100")
    error: Optional[str] = Field(default=None, description="Error message")
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = Field(
        default=None, description="Processing time in milliseconds"
    )
    queue_time_ms: Optional[int] = Field(
        default=None, description="Queue time in milliseconds"
    )


class LLMUsage(BaseModel):
    """LLM Usage model matching the database schema."""

    id: str = Field(description="Usage identifier")
    user_id: Optional[str] = Field(default=None, description="User ID (stored as text)")
    chat_id: Optional[str] = Field(default=None, description="Chat ID (stored as text)")
    request_id: Optional[str] = Field(
        default=None, description="Same as job_id for linking"
    )
    dashboard_id: Optional[str] = Field(
        default=None, description="Dashboard ID for analytics"
    )
    model: str = Field(description="Model name")
    input_tokens: int = Field(description="Input tokens count")
    output_tokens: int = Field(description="Output tokens count")
    total_cost: float = Field(description="Total cost")
    created_at: Optional[datetime] = None


# Input/Output Models for API functions


class ChatMessageInput(BaseModel):
    """Input model for appending chat messages."""

    chat_id: str = Field(description="Chat identifier")
    role: str = Field(description="Message role ('user', 'system', 'assistant')")
    message: str = Field(description="Message content")
    context_widget_ids: Optional[List[str]] = Field(
        default=None, description="Context widget IDs"
    )
    target_widget_type: Optional[str] = Field(
        default=None, description="Target widget type"
    )
    target_chart_sub_type: Optional[str] = Field(
        default=None, description="Chart sub-type"
    )


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


class CreateJobInput(BaseModel):
    """Input model for creating jobs."""

    job_id: str = Field(description="Job identifier")
    user_id: str = Field(description="User ID")
    dashboard_id: str = Field(description="Dashboard ID")
    status: str = Field(default="pending", description="Initial status")
    progress: int = Field(default=0, description="Initial progress")


class UpdateJobInput(BaseModel):
    """Input model for updating job status."""

    job_id: str = Field(description="Job identifier")
    status: str = Field(description="New status")
    progress: Optional[int] = Field(default=None, description="Progress percentage")
    error: Optional[str] = Field(default=None, description="Error message")


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


class ContextWidget(BaseModel):
    """Information about existing widgets in context."""

    widget_id: str
    title: str
    type: str
    config: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None
    sql: Optional[str] = None


class WidgetTask(BaseModel):
    """A task for creating/updating/deleting a widget."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: Literal["CREATE", "UPDATE", "DELETE"]
    widget_type: str = Field(description="line, bar, pie, area, radial, kpi, table")
    widget_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    data_requirements: List[str] = Field(description="Required columns")
    file_id: str
    priority: int = Field(default=1, description="Higher numbers = higher priority")
    depends_on: Optional[List[str]] = Field(
        default=None, description="Other task_ids this depends on"
    )


class DashboardInfo(BaseModel):
    """Dashboard context information."""

    dashboard_id: str
    name: str
    description: Optional[str] = None
    file_ids: List[str]
    existing_widgets: List[Dict[str, Any]] = Field(default_factory=list)


class TopLevelState(BaseModel):
    """Complete state for the chart generation workflow."""

    # Input from frontend
    user_prompt: str = ""
    dashboard_id: str = ""
    context_widget_ids: Optional[List[str]] = None
    chat_id: str = ""
    request_id: str = ""
    user_id: str = ""
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Coordinator outputs
    file_ids: List[str] = Field(default_factory=list)
    available_data_schemas: Dict[str, FileSchema] = Field(
        default_factory=dict, description="file_id: schema mapping"
    )
    available_sample_data: Dict[str, FileSampleData] = Field(
        default_factory=dict, description="file_id: sample mapping"
    )
    context_widgets: List[ContextWidget] = Field(default_factory=list)
    widget_tasks: List[WidgetTask] = Field(default_factory=list)
    task_completion_status: Dict[str, str] = Field(
        default_factory=dict, description="task_id: status mapping"
    )
    created_widget_ids: List[str] = Field(default_factory=list)

    # Dashboard context
    dashboard_info: Optional[DashboardInfo] = None

    # Control flow
    current_step: str = "coordinator"
    should_continue: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Result
    result: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
