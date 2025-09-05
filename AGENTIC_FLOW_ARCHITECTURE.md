# Agentic Flow Architecture Analysis

## Overview

This document provides a comprehensive analysis of the multi-agent system built using LangGraph for dashboard widget creation and management. The system implements a hierarchical supervisor pattern with specialized agent teams working together to process user requests and create data visualizations.

**IMPORTANT**: This architecture strictly follows the CLAUDE.md guidelines for code organization, with clear separation between agent logic, helper functions, and configuration management.

## Core Libraries and Dependencies

### Primary Framework Stack
- **LangGraph (>=0.3.1)**: Core framework for building multi-agent workflows using StateGraph patterns
- **LangChain Core (>=0.3.0)**: Foundation for agent interactions, tools, and message handling
- **LangChain OpenAI (>=0.1.0)**: OpenAI integration for LLM capabilities
- **Pydantic (>=2.0.0)**: Type-safe state management and data validation

### AI and ML Libraries
- **OpenAI (>=1.76.1)**: LLM API for agent intelligence
- **Langfuse (>=2.0.0)**: Prompt management and observability
- **E2B Code Interpreter (>=1.5.0)**: Sandboxed code execution environment

### Data Processing
- **Pandas (>=2.0.0)**: Data manipulation and analysis
- **NumPy (>=1.24.0)**: Numerical computing
- **Supabase (>=2.0.0)**: Database operations and file storage

### Smart Widget Ordering
- **LLM-based Analysis**: Intelligent widget placement using Langfuse prompts
- **Structured Output**: Pydantic schemas for reliable bulk database updates
- **Text-Chart Pairing**: Cohesive narrative flow with explanatory text before charts

### Development and Configuration
- **Python-dotenv (>=1.0.1)**: Environment variable management
- **Requests (>=2.28.0)**: HTTP client for external API calls

## System Architecture

### High-Level Architecture

The system follows a **Hierarchical Supervisor Pattern** with two main levels, strictly adhering to CLAUDE.md guidelines:

1. **Top-Level Supervisor**: Orchestrates overall task flow and delegates to specialized teams
2. **Widget Agent Team**: Specialized team for widget creation, data processing, and validation

**Code Organization** (Following CLAUDE.md):
- **Graph Definition**: `src/agent/graph.py` - ONLY graph structure, no business logic
- **Helper Functions**: `src/actions/` - ALL utility functions centralized here
- **Widget Ordering**: `src/actions/widget_ordering.py` - LLM-based smart ordering system
- **Agent Tools**: `src/agent/agents/{team}/tools/` - ONLY agent-specific tools, NO helper functions
- **Configuration**: `src/config.py` - ALL environment variables and clients

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Graph                                │
│  ┌─────────────────┐    ┌──────────────────────────────────┐ │
│  │ Top Level       │    │        Widget Agent Team         │ │
│  │ Supervisor      │◄──►│         (Subgraph)              │ │
│  │                 │    │                                  │ │
│  └─────────────────┘    └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Widget Agent Team Subgraph Architecture

The Widget Agent Team implements a **Supervisor-Worker Pattern** with intelligent routing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Widget Agent Team Subgraph                           │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │ Widget          │                                                        │
│  │ Supervisor      │                                                        │
│  │ (LLM Router)    │                                                        │
│  └─────────┬───────┘                                                        │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Worker Nodes                                     │ │
│  │                                                                         │ │
│  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │ │    Data     │  │ Validation  │  │ Database    │  │  Text Block     │ │ │
│  │ │    Node     │  │    Node     │  │ Operations  │  │     Node        │ │ │
│  │ │             │  │             │  │    Node     │  │                 │ │ │
│  │ │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────────┐ │ │ │
│  │ │ │fetch_   │ │  │ │LLM-based│ │  │ │CREATE   │ │  │ │fetch_widget│ │ │ │
│  │ │ │data_tool│ │  │ │validator│ │  │ │UPDATE   │ │  │ │_details     │ │ │ │
│  │ │ └─────────┘ │  │ │with     │ │  │ │DELETE   │ │  │ └─────────────┘ │ │ │
│  │ │ ┌─────────┐ │  │ │confidence│ │  │ │         │ │  │ ┌─────────────┐ │ │ │
│  │ │ │generate_│ │  │ │threshold│ │  │ │operations│ │  │ │generate_    │ │ │ │
│  │ │ │code_tool│ │  │ └─────────┘ │  │ └─────────┘ │  │ │text_content │ │ │ │
│  │ │ └─────────┘ │  │             │  │             │  │ └─────────────┘ │ │ │
│  │ │ ┌─────────┐ │  │             │  │             │  │                 │ │ │
│  │ │ │e2b_     │ │  │             │  │             │  │                 │ │ │
│  │ │ │sandbox_ │ │  │             │  │             │  │                 │ │ │
│  │ │ │tool     │ │  │             │  │             │  │                 │ │ │
│  │ │ └─────────┘ │  │             │  │             │  │                 │ │ │
│  │ └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Flow: START → Widget Supervisor → [Worker Node] → Widget Supervisor → END  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Characteristics**:
- **Central Router**: Widget Supervisor uses LLM-based routing decisions
- **Specialized Workers**: Each node handles specific aspects of widget processing
- **Dynamic Flow**: Workers return to supervisor for intelligent next-step decisions
- **Tool Integration**: Each worker has specialized tools for their domain
- **State Persistence**: Unified state management across all nodes

### Graph Structure

#### 1. Top-Level Supervisor Graph

**State Schema**: `SupervisorGraphState`
- **Entry Point**: `START → top_level_supervisor`
- **Conditional Routing**: Uses `should_delegate_to_widget_team()` function
- **Nodes**:
  - `top_level_supervisor`: Main orchestration node
  - `widget_agent_team`: Adapter node for widget team delegation

**Flow Pattern**:
```
START → top_level_supervisor ⟷ widget_agent_team → END
```

#### 2. Widget Agent Team Subgraph

**State Schema**: `WidgetGraphState`
- **Entry Point**: `START → widget_supervisor`
- **Dynamic Routing**: Supervisor makes intelligent routing decisions based on LLM analysis
- **Nodes**:
  - `widget_supervisor`: Central coordinator and router with LLM-based decision making
  - `data`: Data processing and code generation using create_react_agent
  - `validate_data`: Data validation and quality checks with confidence scoring
  - `db_operations_node`: Database CRUD operations (CREATE/UPDATE/DELETE)
  - `text_block_node`: HTML content generation for text widgets

**Flow Pattern**:
```
START → widget_supervisor → [data|validate_data|db_operations_node|text_block_node] 
                    ↑                                    │
                    └────────────────────────────────────┘
                           (Return for next decision)
```

**Routing Logic**:
- **Intelligent Analysis**: Widget supervisor analyzes complete state including task progress, data availability, errors, and previous node outcomes
- **LLM-Based Decisions**: Uses structured output to determine next node based on comprehensive state analysis
- **Dynamic Adaptation**: Flow adapts based on widget type (chart vs text), operation type, and current completion status
- **Error Recovery**: Supervisor can route back to appropriate nodes for retry with error context
- **Completion Detection**: Automatically terminates when database operations complete successfully

## Agentic Nodes and Their Tools

### Top-Level Supervisor Nodes

#### 1. Top-Level Supervisor Node
**Location**: `src/agent/agents/top_level_supervisor/top_level_supervisor.py`

**Agent Type**: `create_react_agent` with structured output

**Tools Available** (Following CLAUDE.md - Tools wrap helper functions from `src/actions/`):
- **`analyze_available_data`**: 
  - Analyzes dashboard data files and schemas
  - **Agent Tool**: Wraps `get_available_data()` from `actions.dashboard`
  - Output: Data summary and file information

- **`plan_widget_tasks`**: 
  - AI-powered task planning using Langfuse prompts
  - Creates `DelegatedTask` objects with structured output
  - Uses: `TaskCreationPlan` schema for validation

- **`execute_widget_tasks`**: 
  - Executes pending widget tasks by invoking widget subgraph
  - Handles task dependency resolution
  - Updates task status and captures results

- **`finalize_created_widgets`**:
  - **Agent Tool**: Integrates smart widget ordering with configuration finalization
  - **Smart Ordering**: Uses LLM analysis via `analyze_and_order_dashboard_widgets()` from `actions.widget_ordering`
  - **Text-Chart Pairing**: Creates cohesive narrative flow where text blocks precede related charts
  - **Fallback Strategy**: Falls back to sequential ordering if LLM ordering fails
  - **Bulk Operations**: Uses `update_widgets_order_and_configuration()` for efficient database updates
  - **Langfuse Integration**: Dynamic prompts with structured output for reliable ordering decisions

- **`finalize_response`**: 
  - Generates final response to user
  - Handles both success and error cases
  - Terminates workflow with `Command(goto=END)`

**State Management**: 
- Uses `TopLevelSupervisorState` with 23+ fields
- Tracks delegated tasks, supervisor status, and execution context
- Supports error handling and retry logic

**Code Organization Compliance**:
- ✅ Agent logic in `src/agent/agents/top_level_supervisor/`
- ✅ Helper functions in `src/actions/`
- ✅ Agent tools only wrap helper functions

### Widget Agent Team Nodes

#### 1. Widget Supervisor Node
**Location**: `src/agent/agents/widget_agent_team/widget_supervisor.py`

**Agent Type**: LLM-based routing agent with structured output

**Key Functions**:
- **State Analysis**: Comprehensive analysis of current widget state
- **Intelligent Routing**: Uses LLM to determine next node based on state
- **Business Rules**: Applies safety constraints and completion checks
- **Error Handling**: Graceful error recovery and status management

**Routing Logic**:
```python
next_node ∈ ["data", "validate_data", "db_operations_node", "text_block_node", "end"]
```

#### 2. Data Processing Node
**Location**: `src/agent/agents/widget_agent_team/data_agent.py`

**Agent Type**: `create_react_agent` with specialized tools

**Tools Available** (Following CLAUDE.md - Agent tools only):
- **`fetch_data_tool`**: 
  - **Agent Tool**: Wraps data fetching from `actions.dashboard`
  - Fetches raw data from file IDs
  - Creates `FileSchema` and `FileSampleData` objects
  - Uses: `get_data_from_file()`, `get_schema_from_file()`, `get_sample_from_file()`

- **`generate_and_execute_python_code_tool`**: 
  - **Agent Tool**: Combines code generation and E2B execution
  - Creates E2B sandbox, generates code using Langfuse prompts
  - Executes code in sandboxed environment with data context
  - Validates output against `ChartConfigSchema`
  - Kills sandbox automatically on completion/error

**Processing Flow**:
```
fetch_data → generate_and_execute_code (E2B sandbox + validation)
```

**Code Organization Compliance**:
- ✅ Agent logic in `src/agent/agents/widget_agent_team/`
- ✅ Data processing helpers in `src/actions/dashboard.py`
- ✅ E2B helpers in `src/actions/e2b_sandbox.py`

#### 3. Validation Node
**Location**: `src/agent/agents/widget_agent_team/validation_agent.py`

**Agent Type**: LLM-based validation with structured output

**Validation Process**:
- Analyzes generated widget configuration against user requirements
- Uses `DataValidationResult` schema for structured feedback
- Confidence-based decision making (80% threshold)
- Detailed error reporting for retry scenarios

**Output Schema**:
```python
class DataValidationResult:
    is_valid: bool
    confidence_level: int (0-100)
    explanation: str
    missing_requirements: Optional[List[str]]
    data_quality_issues: Optional[List[str]]
```

#### 4. Database Operations Node
**Location**: `src/agent/agents/widget_agent_team/database_agent.py`

**Agent Type**: Direct database operations (no LLM)

**Operations Supported** (Following CLAUDE.md - Uses helper functions from `src/actions/`):
- **CREATE**: Creates new widget with `CreateWidgetInput`
- **UPDATE**: Updates existing widget with `UpdateWidgetInput` 
- **DELETE**: Removes widget by ID

**Integration**: Uses helper functions from `actions.dashboard` and `actions.widget_ordering`:
- `create_widget()` - Helper function for widget creation
- `update_widget()` - Helper function for widget updates
- `delete_widget()` - Helper function for widget deletion
- `get_dashboard_widgets_for_ordering()` - Fetches widgets for smart ordering analysis
- `update_widgets_order_and_configuration()` - Bulk updates for widget ordering and configuration
- `analyze_and_order_dashboard_widgets()` - LLM-based intelligent widget placement system

**Code Organization Compliance**:
- ✅ Database agent logic in `src/agent/agents/widget_agent_team/`
- ✅ ALL database operations in `src/actions/dashboard.py`
- ✅ NO direct database access in agent code

#### 5. Text Block Node
**Location**: `src/agent/agents/widget_agent_team/text_block_agent.py`

**Agent Type**: `create_react_agent` for HTML content generation

**Tools Available** (Following CLAUDE.md - Agent tools only):
- **`fetch_widget_details`**: 
  - **Agent Tool**: Wraps widget retrieval from `actions.dashboard`
  - Retrieves referenced widget information
  - Uses: `get_widget_specs()` helper function
- **`generate_text_content`**: 
  - **Agent Tool**: Creates HTML content using Langfuse prompts
  - Generates semantic HTML without inline styles
  - Uses structured output with `TextBlockContentSchema`

**Special Features**:
- Supports widget references via `reference_widget_id`
- Generates semantic HTML without inline styles
- Uses structured output with `TextBlockContentSchema`

**Code Organization Compliance**:
- ✅ Text block agent logic in `src/agent/agents/widget_agent_team/`
- ✅ Widget retrieval helpers in `src/actions/dashboard.py`
- ✅ Prompt management in `src/actions/prompts.py`

## Supported Widget Types and Operations

### Widget Types
The system supports 8 widget types:
```python
widget_type ∈ ["line", "bar", "pie", "area", "radial", "kpi", "table", "text"]
```

### Smart Widget Ordering
The system now includes intelligent widget placement that:
- **Analyzes Relationships**: Uses LLM to understand content relationships between widgets
- **Creates Narrative Flow**: Places explanatory text blocks immediately before their related charts
- **Avoids Grouping**: Prevents poor UX patterns like "all charts first, then all text"
- **Uses Structured Output**: `WidgetOrderingSchema` with `WidgetOrder` objects for reliable bulk updates
- **Langfuse Integration**: Dynamic prompts with variable substitution for consistent ordering decisions

### Operations
Three core operations are supported:
```python
operation ∈ ["CREATE", "UPDATE", "DELETE"]
```

### Data Flow Examples

#### Example 1: Chart Widget Creation Flow

1. **User Request**: "Create a bar chart showing sales by region"

2. **Top-Level Supervisor**:
   ```
   analyze_available_data → plan_widget_tasks → execute_widget_tasks
   ```

3. **Widget Team Execution**:
   ```
   widget_supervisor → data → validate_data → db_operations_node → END
   ```

4. **Data Node Processing**:
   ```
   fetch_data_tool → generate_and_execute_python_code_tool
   ```

5. **State Transitions**:
   ```
   task_status: "pending" → "in_progress" → "completed"
   widget_creation_completed: false → true
   ```

6. **Smart Widget Ordering** (Added in finalize_created_widgets):
   ```
   finalize_created_widgets → analyze_and_order_dashboard_widgets
                           → LLM analysis → bulk database update
   ```

#### Example 2: Text Block Creation Flow

1. **User Request**: "Create a text block explaining the sales chart"

2. **Widget Team Execution**:
   ```
   widget_supervisor → text_block_node → db_operations_node → END
   ```

3. **Text Block Processing**:
   ```
   fetch_widget_details(reference_widget_id) → generate_text_content
   ```

4. **Dependency Resolution**:
   - Uses `reference_widget_id` to fetch chart data
   - Generates explanatory HTML content
   - Creates unified `widget_config` for database persistence

## State Management and Data Flow

### State Schemas

#### TopLevelSupervisorState
- **Purpose**: Orchestration and task delegation
- **Key Fields**: 
  - `user_prompt`, `dashboard_id` (required)
  - `delegated_tasks: List[DelegatedTask]`
  - `supervisor_status: Literal["analyzing", "delegating", "monitoring", "completed", "failed"]`
  - `available_data_summary`, `error_messages`

#### WidgetAgentState  
- **Purpose**: Widget-specific processing
- **Key Fields**:
  - `widget_type`, `operation`, `task_instructions`
  - `raw_file_data`, `generated_code`, `code_execution_result`
  - `widget_config` (unified configuration)
  - `widget_creation_completed`, `data_validated`

#### DelegatedTask
- **Purpose**: Task delegation between supervisor and teams
- **Key Fields**:
  - `target_agent: Literal["widget_agent_team"]`
  - `task_status: Literal["pending", "in_progress", "completed", "failed"]`
  - `widget_type`, `operation`, `file_ids`
  - `task_instructions`, `result`, `error_message`

### Command Pattern Usage

All nodes use LangGraph's `Command` pattern for dynamic routing:

```python
return Command(
    goto="next_node",  # or END
    update={"field": "value"}
)
```

This enables:
- **Dynamic Routing**: Nodes control their own next destinations
- **State Updates**: Atomic state modifications
- **Error Handling**: Graceful failure recovery
- **Conditional Logic**: Context-aware decision making

## Integration Points

### External Services
- **Supabase**: Database operations, file storage
- **OpenAI**: LLM capabilities for all agents
- **E2B**: Sandboxed code execution
- **Langfuse**: Prompt management and observability

### Configuration Management (Following CLAUDE.md)
- **Environment Variables**: ALL managed in `src/config.py` (centralized)
- **External Clients**: ALL initialized in `src/config.py` (Supabase, Langfuse, E2B)
- **Langfuse Integration**: Dynamic prompt compilation via `src/config.py`
- **Model Configuration**: Centralized LLM settings in `src/config.py`
- **Graph Business Logic**: Moved to `src/actions/graph_helpers.py`
- **Smart Ordering**: Widget ordering logic centralized in `src/actions/widget_ordering.py`
- **Prompt Management**: Widget ordering prompts managed via `src/actions/prompts.py`

### Error Handling Strategy
- **Retry Logic**: Tool-specific failure counts and limits
- **Graceful Degradation**: Fallback responses for critical failures
- **State Preservation**: Error context maintained across retries
- **User Feedback**: Detailed error messages for debugging

## Execution Patterns

### Parallel Processing
- Tools within nodes can execute in parallel
- State updates are atomic and conflict-free
- Multiple file processing happens concurrently

### Sequential Workflows
- Inter-node communication follows strict ordering
- State validation at each transition point
- Dependency resolution for complex widget relationships

### Conditional Branching
- LLM-driven routing decisions
- Business rule enforcement
- Dynamic workflow adaptation based on context

This architecture provides a robust, scalable foundation for multi-agent dashboard widget creation with clear separation of concerns, type safety, and comprehensive error handling.

## Smart Widget Ordering System

### Overview
The system now includes an intelligent widget ordering capability that creates cohesive dashboard narratives by analyzing widget relationships and placing explanatory text blocks immediately before their related charts.

### Architecture Components

#### Core Module: `src/actions/widget_ordering.py`
- **`analyze_and_order_dashboard_widgets()`**: Main orchestration function
- **`_get_llm_widget_ordering()`**: LLM analysis with Langfuse integration
- **`_apply_widget_ordering()`**: Bulk database updates

#### Data Models: `src/agent/models.py`
```python
class WidgetOrder(BaseModel):
    """Individual widget order specification for bulk operations."""
    id: str = Field(description="Widget ID to update")
    order: int = Field(description="New order position (1-based indexing)", ge=1)

class WidgetOrderingSchema(BaseModel):
    """Structured output schema for widget ordering decisions."""
    reasoning: str = Field(description="Brief explanation of ordering strategy")
    widget_orders: List[WidgetOrder] = Field(description="List of widgets with order numbers")
```

#### Database Operations: `src/actions/dashboard.py`
- **`get_dashboard_widgets_for_ordering()`**: Fetches widgets with required columns (id, type, order, summary)
- **`update_widgets_order_and_configuration()`**: Bulk updates for widget ordering and configuration

### LLM-Based Analysis Process

1. **Data Preparation**: Fetches existing and new widgets with minimal required data
2. **Langfuse Integration**: Retrieves dynamic prompts with variable substitution
3. **LLM Analysis**: Uses structured output to analyze widget relationships and create ordering strategy
4. **Bulk Operations**: Applies recommendations via efficient database bulk updates
5. **Fallback Strategy**: Falls back to sequential ordering if LLM analysis fails

### Text-Chart Narrative Pairing

The system emphasizes creating **cohesive narrative flow** by:
- **Analyzing Content Relationships**: Using widget summaries to identify explanatory text blocks
- **Sequential Placement**: Placing text blocks immediately before their related charts
- **Avoiding Poor Patterns**: Preventing "all charts first, then all text" layouts
- **Story-like Flow**: Creating dashboards that read like coherent narratives

### Integration Points

#### Finalize Created Widgets Tool
The smart ordering is integrated into the `finalize_created_widgets` tool:
```python
# Primary: Smart LLM-based ordering
result = analyze_and_order_dashboard_widgets(
    dashboard_id=dashboard_id,
    new_widget_ids=widget_ids,
    chat_id=chat_id,
    user_id=user_id
)

# Fallback: Sequential ordering if LLM fails
if not result["success"]:
    fallback_result = update_widgets_configuration_status(
        widget_ids, sequential_order=True
    )
```

#### Langfuse Prompt Management
- **Dynamic Prompts**: Stored in Langfuse at `top_level_supervisor/tools/widget_ordering`
- **Variable Substitution**: Uses {{dashboard_id}}, {{existing_widgets}}, {{new_widgets}} etc.
- **Model Configuration**: Temperature, reasoning effort, and model settings from Langfuse
- **Structured Output**: Enforced via Pydantic schemas for reliable parsing

### Error Handling and Reliability

#### Multi-Level Fallbacks
1. **Primary**: LLM-based smart ordering with relationship analysis
2. **Secondary**: Sequential ordering with configuration updates
3. **Tertiary**: Basic configuration updates without ordering changes

#### Robust Error Recovery
- **SQL Syntax Protection**: Fixed column alias issues in database queries
- **Data Transformation**: Handles database schema differences (type vs widget_type)
- **Langfuse Format Handling**: Properly extracts content from chat message formats
- **Comprehensive Logging**: Detailed debug information for troubleshooting

### Performance Optimizations

#### Efficient Database Operations
- **Minimal Data Fetching**: Only retrieves required columns (id, type, order, summary)
- **Bulk Updates**: Single transaction for multiple widget order updates
- **Individual Updates**: Safer than upsert for existing widget modifications
- **Optimized Queries**: Uses proper column aliases and indexes

#### LLM Efficiency
- **Structured Output**: Reduces parsing overhead and improves reliability
- **Focused Prompts**: Specific instructions for text-chart pairing requirements
- **Model Configuration**: Optimized temperature and reasoning settings
- **Variable Substitution**: Efficient prompt compilation with dynamic data

This smart ordering system transforms the dashboard creation experience from basic widget placement to intelligent narrative construction, ensuring users receive dashboards that tell coherent, easy-to-follow stories through their data visualizations.

## CLAUDE.md Compliance

This architecture has been fully refactored to comply with the CLAUDE.md guidelines:

### ✅ Code Organization Rules Followed

**DO's Implemented:**
- ✅ **Helper Functions**: ALL utility functions moved to `src/actions/` in appropriate files
- ✅ **Agent Nodes**: Each node in separate files for readability
- ✅ **Configuration**: ALL clients and environment variables centralized in `src/config.py`
- ✅ **Graph Structure**: `src/agent/graph.py` contains ONLY graph definition, NO business logic
- ✅ **Agent Tools**: Agent `tools/` folders contain ONLY agent tools that wrap helper functions
- ✅ **Type Safety**: Proper type hints and Pydantic models throughout
- ✅ **Documentation**: Comprehensive docstrings following guidelines

**DON'Ts Eliminated:**
- ❌ **Business Logic in graph.py**: Moved to `src/actions/graph_helpers.py`
- ❌ **Helper Functions in Agent Tools**: Moved to `src/actions/dashboard.py`
- ❌ **Duplicated Database Operations**: Consolidated in `src/actions/dashboard.py`
- ❌ **Direct Database Access**: ALL goes through helper functions
- ❌ **Environment Variables**: ALL centralized in `src/config.py`

### 📁 File Organization Summary

```
src/
├── actions/                    # ✅ ALL helper functions centralized
│   ├── dashboard.py           # ✅ Data & widget operations
│   ├── widget_ordering.py     # ✅ NEW: LLM-based smart widget ordering
│   ├── prompts.py            # ✅ Langfuse prompt management
│   ├── graph_helpers.py       # ✅ Business logic from graph.py
│   ├── database_operations.py # ✅ Database helpers
│   └── ...
├── agent/
│   ├── graph.py              # ✅ ONLY graph structure, no business logic
│   ├── models.py             # ✅ NEW: WidgetOrderingSchema, WidgetOrder
│   └── agents/
│       ├── top_level_supervisor/
│       │   ├── tools/
│       │   │   └── database_operations.py  # ✅ UPDATED: Smart ordering integration
│       │   └── ...
│       └── widget_agent_team/
│           ├── tools/        # ✅ ONLY agent tools (wrap helpers)
│           └── ...
└── config.py                 # ✅ ALL environment variables & clients
```

### 🔄 Key Refactoring Changes

1. **Moved `get_available_data()`** from agent tools to `actions.dashboard.py`
2. **Moved business logic functions** from `graph.py` to `actions.graph_helpers.py`
3. **Centralized Langfuse configuration** in `config.py`
4. **Eliminated duplicate database operations** between agent tools and actions
5. **Updated all agent tools** to wrap helper functions instead of containing business logic
6. **Added smart widget ordering system** in `actions.widget_ordering.py`:
   - LLM-based analysis for optimal widget placement
   - Text-chart narrative pairing to avoid poor UX patterns
   - Structured output with `WidgetOrderingSchema` for reliable database updates
   - Langfuse integration for dynamic prompt management
   - Fallback mechanisms for graceful error handling

This ensures the architecture is maintainable, testable, and follows the strict separation of concerns mandated by CLAUDE.md guidelines.
