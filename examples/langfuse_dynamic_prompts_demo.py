#!/usr/bin/env python3
"""
Demonstration of Dynamic Variable Injection with Langfuse Prompts

This script shows how dynamic variables from LangGraph state can be injected
into Langfuse prompt templates at runtime.
"""

from typing import Dict, Any


def demonstrate_langfuse_variable_injection():
    """Show how Langfuse dynamic variable injection works."""
    
    print("🎯 LANGFUSE DYNAMIC VARIABLE INJECTION DEMO")
    print("=" * 50)
    
    # Example 1: Basic Langfuse prompt template with variables
    print("\n📝 Example Langfuse Prompt Template:")
    langfuse_prompt_template = """
You are a top-level supervisor for dashboard {{dashboard_id}}.

**Current Request:** {{user_prompt}}
**User ID:** {{user_id}}
**Chat ID:** {{chat_id}}
**Status:** {{supervisor_status}}

**Available Data Summary:**
{{available_data_summary}}

**Current Task Instructions:**
{{task_instructions}}

**Delegated Tasks:** {{delegated_tasks}} tasks in progress

Based on the above context, coordinate the workflow to fulfill the user's request.
Start by analyzing available data, then create the appropriate widget tasks.
"""
    
    print(langfuse_prompt_template)
    
    # Example 2: Runtime variables from LangGraph state
    print("\n🔧 Runtime Variables from LangGraph State:")
    runtime_variables = {
        "user_prompt": "Create a sales trend chart showing monthly performance",
        "dashboard_id": "CSpAe9A6GOn5DUCo8sOA0", 
        "chat_id": "chat_789",
        "user_id": "user_456",
        "supervisor_status": "analyzing",
        "available_data_summary": "Sales data with columns: date, revenue, product_category, region. Contains 12 months of data from 2024.",
        "task_instructions": "Analyze sales data and create line chart visualization showing trends over time",
        "delegated_tasks": 2
    }
    
    for key, value in runtime_variables.items():
        print(f"  {key}: {value}")
    
    # Example 3: Show what happens after Langfuse .compile()
    print("\n✨ After Langfuse .compile() with Runtime Variables:")
    compiled_prompt = langfuse_prompt_template
    for key, value in runtime_variables.items():
        compiled_prompt = compiled_prompt.replace(f"{{{{{key}}}}}", str(value))
    
    print(compiled_prompt)
    
    print("\n🎉 DYNAMIC INJECTION BENEFITS:")
    print("✅ Prompts adapt to current LangGraph state automatically")
    print("✅ No hardcoded values - everything is contextual") 
    print("✅ Easy prompt management through Langfuse UI")
    print("✅ Version control and A/B testing of prompts")
    print("✅ Runtime compilation ensures fresh data every time")
    
    # Example 4: Show variable types supported
    print("\n📊 Types of Variables You Can Use:")
    variable_types = {
        "String variables": ["user_prompt", "dashboard_id", "chat_id"],
        "Status variables": ["supervisor_status", "task_status"],
        "Data summaries": ["available_data_summary", "file_schemas"],
        "Counts/Numbers": ["delegated_tasks", "remaining_steps", "iteration_count"],
        "Lists/Arrays": ["available_files", "context_widget_ids", "error_messages"],
        "Datetime fields": ["created_at", "updated_at", "started_at"]
    }
    
    for category, examples in variable_types.items():
        print(f"  {category}: {', '.join(examples)}")
    
    print("\n🚀 IMPLEMENTATION STATUS:")
    print("✅ Langfuse integration completed")
    print("✅ Dynamic variable extraction implemented") 
    print("✅ Runtime compilation system ready")
    print("✅ Error handling with strict requirements")
    print("✅ Support for all LangGraph state variables")


if __name__ == "__main__":
    demonstrate_langfuse_variable_injection()