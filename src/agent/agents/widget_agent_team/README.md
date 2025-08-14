# Widget Agent Team

A LangGraph-based system for processing data visualization tasks using a supervisor pattern with intelligent routing to specialized worker nodes.

## Overview

The Widget Agent Team implements a sophisticated multi-agent system that can:
- Process data visualization requests
- Generate Python code for data manipulation
- Execute code safely in a sandboxed environment
- Validate output data for specific widget types
- Handle retry logic with intelligent error recovery

## Architecture

### Core Components

#### 1. **WidgetSupervisor** (`widget_supervisor.py`)
- Central coordinator that analyzes complete state
- Uses LLM to make intelligent routing decisions
- Applies business logic constraints (max retries, validation rules)
- Routes tasks to appropriate worker nodes

#### 2. **Worker Nodes** (`worker_nodes.py`)
- **data_node**: Unified data processing node that fetches data, generates code, and executes it
- **validate_data_node**: LLM-based validation that analyzes data against user requirements with confidence scoring

#### 3. **State Management** (`models.py`)
- **WidgetAgentState**: Complete Pydantic state schema
- **SupervisorDecision**: Structured LLM outputs for routing
- Full type safety with comprehensive state tracking

#### 4. **Tools** (`tools/`)
- **DataProcessor**: Helper functions for data transformation
- **CodeGenerator**: Template generation for different widget types

## Supported Widget Types

- **line**: Line charts (requires x/y arrays)
- **bar**: Bar charts (requires categories/values arrays)  
- **pie**: Pie charts (requires array of label/value objects)
- **table**: Data tables (requires array of objects)
- **kpi**: Key performance indicators (requires value object)
- **area**: Area charts (same as line charts)
- **radial**: Radial charts (same as bar charts)

## Usage

### Basic Usage

```python
from agent.agents.widget_agent_team import WidgetAgentRunner, create_custom_widget

# Create a custom widget
result = await create_custom_widget(
    user_prompt="Create a bar chart showing sales by region",
    widget_type="bar",
    file_ids=["file123"],
    title="Sales by Region",
    description="Bar chart showing sales data"
)

print(f"Status: {result.task_status}")
print(f"Data: {result.data}")
```

### Advanced Usage

```python
from agent.agents.widget_agent_team import WidgetAgentRunner
from agent.models import WidgetAgentState, FileSchema, FileSampleData

runner = WidgetAgentRunner()

# Create detailed task
task = WidgetAgentState(
    task_instructions="Create visualization based on user request",
    user_prompt="Show me a pie chart of customer distribution by region",
    operation="CREATE",
    widget_type="pie",
    file_ids=["customers.csv"],
    title="Customer Distribution",
    description="Pie chart showing customers by region",
    # Add file schemas and sample data...
)

# Process through the agent system
result = await runner.process_widget_task(task)
```

## Workflow

1. **START** → **widget_supervisor**
2. **widget_supervisor** analyzes complete state and routes to:
   - **fetch_data** (if no raw data exists)
   - **generate_code** (if code needs generation/regeneration)
   - **execute_code** (if code exists but not executed)
   - **validate_data** (if execution successful but not validated)
   - **update_task** (to finalize with success/failure)
   - **END** (if task complete)
3. All worker nodes report back to **widget_supervisor**
4. **widget_supervisor** makes next routing decision
5. Process continues until task completion or max retries reached

## Key Features

### Intelligent Routing
- LLM-powered supervisor analyzes complete state context
- Business logic constraints (max 3 iterations)
- Error recovery strategies
- Context-aware decision making

### Safe Code Execution
- Sandboxed Python environment
- Limited built-ins for security
- Automatic pandas/numpy imports when available
- Clear error reporting with stack traces

### Widget-Specific Validation
- Custom validators for each widget type
- Data structure verification
- Clear validation error messages
- Automatic retry on validation failure

### Comprehensive State Tracking
- Full workflow history
- Error message accumulation
- Iteration counting
- Processing time metrics
- Reasoning logs from supervisor

## Configuration

The system uses OpenAI GPT-4o-mini by default but can be configured:

```python
from agent.agents.widget_agent_team.widget_supervisor import WidgetSupervisor
from agent.agents.widget_agent_team.worker_nodes import WorkerNodes

# Custom LLM configuration
supervisor = WidgetSupervisor(llm_model="openai:gpt-4")
workers = WorkerNodes(llm_model="openai:gpt-4")
```

## Error Handling

- Graceful error recovery with retry logic
- Maximum 3 iterations to prevent infinite loops
- Detailed error logging and reporting
- State preservation across failures
- Automatic fallback to sample data when files unavailable

## Integration

The Widget Agent Team integrates with:
- **Supabase** for file storage and retrieval
- **LangGraph** for workflow orchestration
- **Pydantic** for type safety and validation
- **pandas/numpy** for data processing
- **OpenAI** for code generation

## Examples

See `main.py` for comprehensive usage examples including:
- Bar charts with file data
- Pie charts without file data  
- KPI widgets
- Table widgets
- Error handling scenarios

## Development

To extend the system:

1. **Add new widget types**: Update validators in `validate_data_node`
2. **Add new worker nodes**: Create in `worker_nodes.py` and register in graph
3. **Modify routing logic**: Update supervisor decision prompts
4. **Add new tools**: Create in `tools/` directory

## Testing

Run the development setup and tests:

```bash
make dev
make test
make format
```