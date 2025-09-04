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
  - **Agent Tool**: Wraps `update_widgets_configuration_status()` from `actions.dashboard`
  - Finalizes widgets created by completed tasks
  - Marks widgets as configured (live)

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

**Integration**: Uses helper functions from `actions.dashboard`:
- `create_widget()` - Helper function for widget creation
- `update_widget()` - Helper function for widget updates
- `delete_widget()` - Helper function for widget deletion

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
│   ├── graph_helpers.py       # ✅ Business logic from graph.py
│   ├── database_operations.py # ✅ Database helpers
│   └── ...
├── agent/
│   ├── graph.py              # ✅ ONLY graph structure, no business logic
│   └── agents/
│       ├── top_level_supervisor/
│       │   ├── tools/        # ✅ ONLY agent tools (wrap helpers)
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

This ensures the architecture is maintainable, testable, and follows the strict separation of concerns mandated by CLAUDE.md guidelines.
