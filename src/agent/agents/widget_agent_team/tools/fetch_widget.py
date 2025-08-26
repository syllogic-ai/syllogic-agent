"""Tool to fetch widget details by widget_id."""

import logging
from typing import Dict, Any

from agent.models import WidgetAgentState

# Import config using the robust import utility
def _get_supabase_client():
    """Get Supabase client using robust import."""
    try:
        # Try direct import first
        from config import get_supabase_client
        return get_supabase_client()
    except ImportError:
        # Fallback to robust import from utils
        import sys
        import os
        # Add the src directory to the path
        src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from actions.utils import import_config
        config_module = import_config()
        return config_module.get_supabase_client()

logger = logging.getLogger(__name__)


def fetch_widget_details(widget_id: str) -> Dict[str, Any]:
    """Fetch widget details including config from database.
    
    Args:
        widget_id: Widget identifier
        
    Returns:
        Dict containing widget details with config, title, description, etc.
        
    Raises:
        Exception: If widget not found or fetch fails
    """
    try:
        supabase = _get_supabase_client()
        
        # Fetch widget details from database
        widget_result = (
            supabase.table("widgets")
            .select("id, title, type, config, data, sql, dashboard_id")
            .eq("id", widget_id)
            .single()
            .execute()
        )
        
        if not widget_result.data:
            raise Exception(f"Widget {widget_id} not found")
            
        widget_data = widget_result.data
        
        logger.info(f"Successfully fetched widget details for {widget_id}")
        
        return {
            "widget_id": widget_data["id"],
            "title": widget_data.get("title", ""),
            "description": "",  # Description not in database schema - use empty string
            "widget_type": widget_data.get("type", ""),  # Field is 'type' not 'widget_type'
            "config": widget_data.get("config", {}),
            "data": widget_data.get("data", {}),
            "sql": widget_data.get("sql", ""),
            "dashboard_id": widget_data.get("dashboard_id", "")
        }
        
    except Exception as e:
        error_msg = f"Failed to fetch widget details for {widget_id}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def fetch_widget_tool(state: WidgetAgentState, tool_call_id: str) -> str:
    """Tool function to fetch widget details for text block generation.
    
    Args:
        state: Current widget agent state
        tool_call_id: Tool call identifier
        
    Returns:
        String result for LLM consumption
    """
    try:
        # Use reference_widget_id for text blocks analyzing existing widgets
        target_widget_id = state.reference_widget_id if state.reference_widget_id else state.widget_id
        
        if not target_widget_id:
            return "ERROR: No reference_widget_id or widget_id provided in state"
            
        widget_details = fetch_widget_details(target_widget_id)
        
        # Format for LLM consumption
        result = f"""Referenced Widget Details Retrieved:
- Widget ID: {widget_details['widget_id']}
- Title: {widget_details['title']}
- Description: {widget_details['description']}
- Type: {widget_details['widget_type']}
- Dashboard ID: {widget_details['dashboard_id']}
- Config: {widget_details['config']}
- Current Data: {widget_details['data']}
"""
        
        logger.info(f"fetch_widget_tool completed for reference_widget_id: {target_widget_id}")
        return result
        
    except Exception as e:
        error_msg = f"fetch_widget_tool failed: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"