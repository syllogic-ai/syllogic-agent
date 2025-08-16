# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this LangGraph-based agentic system. This documentation follows a logical flow from understanding the architecture to development practices and testing standards.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Development Standards](#development-standards)
5. [Testing Requirements](#testing-requirements)
6. [Development Workflow](#development-workflow)
7. [Code Organization Rules](#code-organization-rules)
8. [Integration Patterns](#integration-patterns)

## Quick Start

### Initial Setup
```bash
# Complete project setup with all dependencies
make setup

# Or step by step:
make dev          # Install with development dependencies
make serve        # Start LangGraph development server
```

### Development Commands
```bash
# Testing
make test                                    # Run all unit tests
make test TEST_FILE=tests/unit_tests/test_chats.py  # Run specific test file
make integration_tests                       # Run integration tests
make test_watch                             # Run tests in watch mode (auto-reload)
make extended_tests                         # Run extended tests

# Code Quality
make format                                 # Format code
make lint                                   # Lint code
make spell_check                           # Check spelling

# LangGraph Development
make serve                                  # Start LangGraph dev server
python -m pytest tests/integration_tests/test_graph.py  # Run graph directly
```

## Architecture Overview

This is a **LangGraph-based agentic system** designed for scalable dashboard/widget management with conversational AI integration. The architecture follows a strict separation of concerns with dedicated modules for different responsibilities.

### Core Design Principles

1. **LangGraph-First**: All agentic workflows are built using LangGraph's StateGraph pattern
2. **Separation of Concerns**: Clear distinction between agent logic, helper functions, and configuration
3. **Agent Team Organization**: Each agent team lives in its dedicated folder with specialized tools
4. **Helper Function Centralization**: All utility functions reside in `src/actions/`
5. **Configuration Management**: All environment variables and clients initialized in `config.py`
6. **Test-Driven Development**: Every function must have corresponding unit tests

## Core Components

### 1. LangGraph Agent System (`src/agent/`)

#### Graph Definition (`src/agent/graph.py`)
- **Purpose**: Defines and builds the LangGraph StateGraph
- **Responsibility**: ONLY graph creation, state definition, and compilation
- **Exports**: `graph` object for LangGraph server
- **State Management**: Uses `WidgetAgentState` from `models.py`
- **Context Schema**: `Context` TypedDict for runtime configuration

**Key Rules for `graph.py`:**
- NO business logic implementation
- NO helper functions
- NO direct database operations
- ONLY graph structure and state management

#### State Models (`src/agent/models.py`)
- Comprehensive Pydantic models for all system entities
- Type-safe state management with proper validation
- Models include: `WidgetAgentState`, `Chat`, `Widget`, `Job`, etc.

### 2. Agent Teams (`src/agent/agents/`)

Each agent team follows this strict structure:

```
src/agent/agents/{team_name}/
├── __init__.py              # Team exports
├── main.py                  # Entry point and runner
├── {supervisor_name}.py     # Supervisor node (routing logic)
├── {worker_name}.py         # Individual worker nodes (separate files)
├── {agent_name}.py          # Specialized agent implementations
└── tools/                   # Agent-specific tools ONLY
    ├── __init__.py
    ├── {tool_name}.py       # Agent tools (NOT helper functions)
    └── ...
```

#### Current Agent Teams

**Widget Agent Team** (`src/agent/agents/widget_agent_team/`)
- **Supervisor**: `widget_supervisor.py` - Central coordinator and task router
- **Workers**: Individual nodes in separate files (`worker_nodes.py`, `data_agent.py`, etc.)
- **Tools**: Agent-specific tools in `tools/` folder
- **Main**: `main.py` - Entry point with `WidgetAgentRunner`

**Agent Team Rules:**
- Each node MUST be in a separate file for readability
- Tools folder contains ONLY agent tools, never helper functions
- Supervisor handles routing decisions and state management
- Workers perform specific tasks and return to supervisor

### 3. Helper Functions (`src/actions/`)

All utility and helper functions MUST reside here, organized by domain:

#### Current Action Modules

**`chats.py`** - Chat Management
- `append_chat_message()` - Appends messages to conversation arrays
- `get_message_history()` - Retrieves chat conversation history
- Uses JSONB conversation arrays with timestamps and roles

**`dashboard.py`** - File and Widget Operations
- File operations: `get_data_from_file()`, `get_schema_from_file()`, `get_sample_from_file()`
- Widget CRUD: `create_widget()`, `update_widget()`, `delete_widget()`, `get_widget_specs()`
- Supports CSV, Excel, JSON file formats via pandas
- Integrates with Supabase Storage for file retrieval

**`jobs.py`** - Async Job Management
- Job lifecycle: `create_job()` → `processing_job()` → `complete_job()`/`fail_job()`
- Status tracking: "pending" → "processing" → "completed"/"failed"
- Automatic timing calculations (processing time, queue time)
- Cleanup utilities for old jobs

**`utils.py`** - Data Transformation Utilities
- Chart data conversion: `convert_data_to_chart_data()`, `convert_data_to_chart_data_1d()`
- Value sanitization: `convert_value()` handles pandas/numpy types
- Configuration helpers: `convert_chart_data_to_chart_config()`
- JSON cleaning: `remove_null_pairs()` recursively removes None values
- State reducers: `take_last()`, `merge_lists()` for LangGraph concurrent updates

### 4. Configuration Management (`src/config.py`)

**Purpose**: Initialize all environmental variables, clients, and shared resources

**Key Functions:**
- `get_supabase_client()` - Singleton Supabase client with proper error handling
- `get_e2b_api_key()` - E2B API key management
- `create_e2b_sandbox()` - E2B Sandbox creation
- Reset functions for testing: `reset_supabase_client()`, `reset_e2b_config()`

**Configuration Rules:**
- ALL environment variables MUST be handled here
- ALL external clients MUST be initialized here
- Proper error handling with logging
- Singleton patterns for resource efficiency
- Test-friendly reset functions

## Development Standards

### File Organization Rules

1. **Helper Functions**: ALL utility functions go in `src/actions/` in the most appropriate file
2. **Agent Logic**: Stays in `src/agent/` with proper team organization
3. **Graph Definition**: ONLY in `src/agent/graph.py` - no business logic
4. **Configuration**: ALL environment setup in `src/config.py`
5. **Agent Teams**: Each team in dedicated folder with separate node files
6. **Agent Tools**: Only in team's `tools/` folder - never helper functions

### Code Quality Standards

1. **Type Safety**: Extensive use of TypedDict, dataclasses, and type hints
2. **Error Handling**: All functions use try/catch with structured logging
3. **Documentation**: Comprehensive docstrings following Google style
4. **Async Support**: Both sync and async function variants where needed
5. **Data Validation**: Input validation and safe data conversion throughout

### Import Management

**Critical Import Rules:**
- Always check and test imports when creating/editing files
- Use absolute imports where possible
- Handle circular dependencies properly (see `dashboard.py` example)
- Import from `config.py` for all clients and environment variables

## Testing Requirements

### Unit Testing Standards

**For Helper Functions** (`src/actions/`):
1. Create unit test in `tests/unit_tests/test_{module_name}.py`
2. Test ALL functions with comprehensive coverage
3. Use mocked Supabase clients and external dependencies
4. Test both success and error scenarios
5. Verify test passes before keeping the implementation

**For Non-Helper Functions** (Agent tools, temporary utilities):
1. Create test file to verify implementation
2. Run test to ensure it passes
3. **DELETE the test file after verification** (temporary validation only)

### Testing Commands
```bash
# Run all unit tests
make test

# Run specific test module
make test TEST_FILE=tests/unit_tests/test_chats.py

# Run integration tests (end-to-end workflows)
make integration_tests

# Watch mode for development
make test_watch
```

### Test Structure
```
tests/
├── conftest.py              # Global test configuration
├── unit_tests/              # Unit tests for helper functions
│   ├── conftest.py         # Unit test fixtures
│   ├── test_chats.py       # Chat functionality tests
│   ├── test_dashboard.py   # Dashboard operations tests
│   ├── test_jobs.py        # Job management tests
│   └── test_utils.py       # Utility function tests
└── integration_tests/       # End-to-end workflow tests
    ├── __init__.py
    └── test_graph.py       # LangGraph integration tests
```

## Development Workflow

### 1. Creating New Helper Functions
```bash
# Step 1: Add function to appropriate file in src/actions/
# Step 2: Create/update unit test in tests/unit_tests/
# Step 3: Run test to verify implementation
make test TEST_FILE=tests/unit_tests/test_{module}.py
# Step 4: If test passes, keep both function and test
```

### 2. Creating New Agent Teams
```bash
# Step 1: Create team folder in src/agent/agents/{team_name}/
# Step 2: Implement supervisor, workers, and tools in separate files
# Step 3: Create main.py entry point
# Step 4: Update graph.py to include new nodes
# Step 5: Create integration test for the team workflow
```

### 3. Adding New Dependencies
```bash
# Step 1: Add dependency to pyproject.toml
# Step 2: Update relevant imports
# Step 3: Test all affected modules
make test
make integration_tests
```

### 4. Modifying Graph Structure
```bash
# Step 1: Update graph.py ONLY for structure changes
# Step 2: Ensure all business logic remains in agent files
# Step 3: Run integration tests
make integration_tests
```

## Code Organization Rules

### DO's ✅

- **Helper Functions**: Always place in `src/actions/` in the most appropriate file
- **Agent Nodes**: Keep each node in a separate file for readability
- **Configuration**: Initialize all clients and environment variables in `config.py`
- **Testing**: Create unit tests for all helper functions
- **Type Safety**: Use proper type hints and Pydantic models
- **Documentation**: Write comprehensive docstrings
- **Import Testing**: Always verify imports work correctly

### DON'Ts ❌

- **Graph File**: Never put business logic in `src/agent/graph.py`
- **Agent Tools**: Never put helper functions in agent `tools/` folders
- **Circular Imports**: Avoid by using proper import patterns
- **Missing Tests**: Never commit helper functions without unit tests
- **Direct Database Access**: Always use helper functions from `src/actions/`
- **Environment Variables**: Never initialize outside of `config.py`

## Integration Patterns

### Database Operations
All database operations flow through `src/actions/` helper functions:

1. **Supabase Client**: Retrieved via `config.get_supabase_client()`
2. **CRUD Operations**: Implemented in appropriate action modules
3. **Error Handling**: Consistent error handling with logging
4. **Type Safety**: Using Pydantic models for all data structures

### LangGraph Integration
- **State Management**: `WidgetAgentState` handles all workflow state
- **Node Communication**: Nodes communicate through state updates
- **Supervisor Pattern**: Central supervisor routes tasks to specialized workers
- **Error Recovery**: Proper error handling and retry mechanisms

### Data Flow Architecture
```
Input → LangGraph State → Supervisor Node → Worker Nodes → Helper Functions → Database
                     ↑                                            ↓
                     └── State Updates ←── Response Processing ←──┘
```

This architecture ensures scalable, maintainable, and testable agentic workflows with clear separation of concerns and robust error handling throughout the system.