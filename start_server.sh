#!/bin/bash

# Start LangGraph server using the virtual environment
# This ensures all dependencies including langfuse are available

echo "🚀 Starting LangGraph server with Langfuse integration..."
echo "Using virtual environment Python and langgraph CLI"

# Activate virtual environment and start server
source .venv/bin/activate

# Set PYTHONPATH to include src directory
export PYTHONPATH=src:$PYTHONPATH

# Start the langgraph development server
.venv/bin/langgraph dev --host localhost --port 2024

echo "Server stopped."