# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Initial Setup
```bash
# Complete project setup with all dependencies
make setup

# Or step by step:
make dev          # Install with development dependencies
make serve        # Start LangGraph development server
```

### Testing
```bash
# Run all unit tests
make test

# Run specific test file
make test TEST_FILE=tests/unit_tests/test_chats.py

# Run integration tests
make integration_tests

# Run tests in watch mode (auto-reload on changes)
make test_watch

# Run extended tests
make extended_tests
```

### Code Quality
```bash
# Format code
make format

# Lint code
make lint

# Check spelling
make spell_check
```

### LangGraph Development
```bash
# Start LangGraph dev server (after setup)
make serve

# Run the agent graph directly
python -m pytest tests/integration_tests/test_graph.py
```

## Architecture Overview

This is a **LangGraph-based agent system** designed to work with Supabase databases and provide utilities for dashboard/widget management, chat operations, and job processing.

### Core Components

#### 1. LangGraph Agent (`src/agent/graph.py`)
- **Single-node template** using StateGraph with configurable runtime context
- Entry point: `graph` object exported from the module
- Context schema: `Context` TypedDict with `my_configurable_param`
- State management: `State` dataclass with `changeme` field
- Main processing function: `call_model()` - processes input and returns configured output

#### 2. Action Modules (`src/actions/`)
- **`chats.py`**: Chat message management for Supabase `chats` table
  - `append_chat_message()` - appends messages to conversation arrays
  - `get_message_history()` - retrieves chat conversation history
  - Uses JSONB conversation arrays with timestamps and roles

- **`dashboard.py`**: File and widget operations for dashboards
  - File operations: `get_data_from_file()`, `get_schema_from_file()`, `get_sample_from_file()`
  - Widget CRUD: `create_widget()`, `update_widget()`, `delete_widget()`, `get_widget_specs()`
  - Supports CSV, Excel, JSON file formats via pandas
  - Integrates with Supabase Storage for file retrieval

- **`jobs.py`**: Async job management with status tracking
  - Job lifecycle: `create_job()` → `processing_job()` → `complete_job()`/`fail_job()`
  - Status tracking: "pending" → "processing" → "completed"/"failed"
  - Automatic timing calculations (processing time, queue time)
  - Cleanup utilities for old jobs

- **`utils.py`**: Data transformation utilities
  - Chart data conversion: `convert_data_to_chart_data()`, `convert_data_to_chart_data_1d()`
  - Value sanitization: `convert_value()` handles pandas/numpy types
  - Configuration helpers: `convert_chart_data_to_chart_config()`
  - JSON cleaning: `remove_null_pairs()` recursively removes None values

### Database Integration

The system is built around **Supabase** as the primary database:
- **Users table**: Basic user management
- **Dashboards table**: Dashboard metadata and configuration
- **Files table**: Uploaded file tracking with storage paths
- **Chats table**: Conversation management with JSONB arrays
- **Widgets table**: Dashboard widget specifications and data
- **Jobs table**: Async task tracking and status management
- **LLM Usage table**: AI usage and cost tracking

### Key Patterns

1. **Error Handling**: All functions use try/catch with structured logging
2. **Type Safety**: Extensive use of TypedDict, dataclasses, and type hints
3. **Async Support**: Both sync and async function variants where needed
4. **Data Validation**: Input validation and safe data conversion throughout
5. **Configuration Management**: Uses `.env` files with environment variable fallbacks

### Development Workflow

1. **Graph Development**: Modify `src/agent/graph.py` for core logic changes
2. **Action Extensions**: Add new database operations to respective action modules
3. **Testing**: Unit tests in `tests/unit_tests/` with mock Supabase clients
4. **Integration**: Integration tests in `tests/integration_tests/` for end-to-end flows
5. **Configuration**: Use `.env` files for secrets (LangSmith API keys, Supabase credentials)

### Data Flow

1. **Input Processing**: LangGraph receives state through `call_model()`
2. **Database Operations**: Actions modules handle CRUD operations via Supabase client
3. **Data Transformation**: Utils handle pandas/numpy data conversion for charts
4. **Job Management**: Long-running operations tracked via jobs table
5. **Chat Integration**: User interactions stored as conversation arrays

This architecture enables scalable dashboard/widget management with conversational AI integration, built on modern Python async patterns and robust data handling.