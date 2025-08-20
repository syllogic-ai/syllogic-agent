"""Setup script for creating Langfuse prompts.

This script helps create and manage prompts in Langfuse for the agent system.
Run this once with valid Langfuse credentials to set up your prompts.
"""

import logging

# Handle imports for different execution contexts
try:
    from actions.prompts import create_prompt, list_prompts
    from config import get_langfuse_client
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.prompts import create_prompt, list_prompts
    from config import get_langfuse_client

logger = logging.getLogger(__name__)


def setup_top_level_supervisor_prompt():
    """Create the top_level_supervisor prompt in Langfuse."""
    
    # Enhanced system prompt for the top-level supervisor
    system_prompt = """You are a top-level supervisor for an intelligent dashboard management system.

Your role is to orchestrate the entire workflow from user request to widget creation:

## PRIMARY RESPONSIBILITIES:
1. **Data Analysis**: Understand available data sources and their structure
2. **Request Analysis**: Parse user requests for dashboard widgets and visualizations  
3. **Task Planning**: Create intelligent, minimal task plans for widget creation
4. **Task Execution**: Delegate tasks to specialized agent teams
5. **Progress Monitoring**: Track task completion and handle errors
6. **User Communication**: Provide clear status updates and final responses

## WORKFLOW PROCESS:
1. START → analyze_available_data (understand data landscape)
2. PLAN → plan_widget_tasks (AI-powered task creation)  
3. EXECUTE → execute_widget_tasks (delegate to widget_agent_team)
4. MONITOR → check_task_status (track progress)
5. COMPLETE → finalize_response (user-friendly completion)

## AVAILABLE TOOLS:
- **analyze_available_data**: Read and understand available data files and schemas
- **plan_widget_tasks**: Use AI to create intelligent, minimal widget task plans
- **execute_widget_tasks**: Execute planned tasks by delegating to widget_agent_team
- **check_task_status**: Monitor progress of all delegated tasks
- **generate_error_response**: Handle errors with user-friendly messages
- **finalize_response**: Complete workflow when all tasks are successful

## KEY PRINCIPLES:
- **Quality over Quantity**: Create only essential widgets that directly answer user requests
- **Data-Driven**: Base all decisions on actual available data and schemas
- **User-Centric**: Always prioritize clear, helpful responses
- **Error Resilience**: Handle failures gracefully with informative error messages
- **Minimal Planning**: Avoid over-engineering - create only what's specifically requested

## WORKFLOW EXAMPLE:
User: "Show me sales trends for last month"
1. analyze_available_data → Find sales data files
2. plan_widget_tasks → Create 1 line chart task (minimal, focused)
3. execute_widget_tasks → Delegate to widget_agent_team
4. check_task_status → Verify completion
5. finalize_response → "Sales trend chart created successfully"

Always start with data analysis, then proceed systematically through the workflow."""

    # Configuration with model and temperature
    config = {
        "model": "gpt-4o-mini",  # Can be updated to gpt-5-mini when available
        "temperature": 0.2,
        "max_tokens": 4000,
        "response_format": "text"
    }
    
    try:
        # Create the prompt with latest label (production ready)
        created_prompt = create_prompt(
            name="top_level_supervisor",
            prompt=system_prompt,
            config=config,
            labels=["latest", "production"],
            is_active=True
        )
        
        print("✅ Top-level supervisor prompt created successfully!")
        print(f"Prompt name: top_level_supervisor")
        print(f"Labels: ['latest', 'production']")
        print(f"Model: {config['model']}")
        print(f"Temperature: {config['temperature']}")
        
        return created_prompt
        
    except Exception as e:
        print(f"❌ Failed to create prompt: {e}")
        print("Make sure you have valid LANGFUSE credentials in your .env file")
        return None


def setup_widget_supervisor_prompt():
    """Create a widget supervisor prompt as an example."""
    
    widget_prompt = """You are a widget supervisor responsible for creating, updating, and managing dashboard widgets.

Your responsibilities:
- Analyze data requirements for widget creation
- Coordinate with data agents for data processing
- Manage widget lifecycle (create, update, delete)
- Ensure data quality and validation
- Handle widget-specific errors and edge cases

Key considerations:
- Always validate data before widget creation
- Choose appropriate chart types based on data structure
- Provide clear error messages when data is insufficient
- Optimize for performance and user experience"""

    config = {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    try:
        created_prompt = create_prompt(
            name="widget_supervisor",
            prompt=widget_prompt,
            config=config,
            labels=["latest"],
            is_active=True
        )
        
        print("✅ Widget supervisor prompt created successfully!")
        return created_prompt
        
    except Exception as e:
        print(f"❌ Failed to create widget supervisor prompt: {e}")
        return None


def list_existing_prompts():
    """List all existing prompts in Langfuse."""
    try:
        prompts = list_prompts()
        
        if not prompts:
            print("No prompts found in Langfuse.")
            return
            
        print(f"\n📋 Found {len(prompts)} prompts in Langfuse:")
        for prompt in prompts:
            print(f"• {prompt.name} (versions: {prompt.versions})")
            
    except Exception as e:
        print(f"❌ Failed to list prompts: {e}")


def main():
    """Main setup function."""
    print("🔧 Setting up Langfuse prompts for agent system...")
    
    # Check Langfuse connection
    try:
        client = get_langfuse_client()
        if client.auth_check():
            print("✅ Langfuse connection verified")
        else:
            print("⚠️ Langfuse connection issue - some operations may fail")
    except Exception as e:
        print(f"❌ Cannot connect to Langfuse: {e}")
        print("Please check your LANGFUSE credentials in .env file")
        return
    
    print("\n1. Creating top-level supervisor prompt...")
    setup_top_level_supervisor_prompt()
    
    print("\n2. Creating widget supervisor prompt...")  
    setup_widget_supervisor_prompt()
    
    print("\n3. Listing all prompts...")
    list_existing_prompts()
    
    print("\n🎉 Langfuse prompt setup complete!")
    print("\nNow your agents will use Langfuse-managed prompts with fallback support.")


if __name__ == "__main__":
    main()