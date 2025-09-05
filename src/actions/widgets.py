"""Widget management helper functions.

This module provides utilities for CRUD operations on widgets,
including fetching widget configurations and details by ID.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from config import get_supabase_client

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def get_widget_by_id(widget_id: str) -> Optional[Dict[str, Any]]:
    """Get widget details by widget ID.
    
    Args:
        widget_id: The widget ID to fetch
        
    Returns:
        Dict containing widget details including:
        - id: Widget ID
        - dashboard_id: Dashboard ID the widget belongs to
        - title: Widget title
        - type: Widget type ('text', 'chart', 'kpi', 'table')
        - config: Widget configuration (JSON)
        - data: Widget data (JSON)
        - sql: SQL query if applicable
        - layout: Layout configuration
        - order: Display order
        - chat_id: Chat ID if created from chat
        - is_configured: Whether widget is fully configured
        - cache_key: Cache key for data
        - last_data_fetch: Last data fetch timestamp
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        
    Returns None if widget not found.
        
    Raises:
        Exception: If database query fails
    """
    try:
        supabase = get_supabase_client()
        
        logger.info(f"Fetching widget by ID: {widget_id}")
        
        # Query the widgets table
        result = supabase.table("widgets").select("*").eq("id", widget_id).execute()
        
        if not result.data:
            logger.warning(f"Widget not found: {widget_id}")
            return None
            
        widget = result.data[0]
        logger.info(f"Successfully fetched widget: {widget_id} - {widget['title']} ({widget['type']})")
        
        return widget
        
    except Exception as e:
        error_msg = f"Failed to fetch widget {widget_id}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def get_widget_config_by_id(widget_id: str) -> Optional[Dict[str, Any]]:
    """Get widget configuration by widget ID.
    
    Args:
        widget_id: The widget ID to fetch configuration for
        
    Returns:
        Dict containing widget configuration or None if widget not found
        
    Raises:
        Exception: If database query fails
    """
    try:
        widget = get_widget_by_id(widget_id)
        
        if not widget:
            return None
            
        return widget.get('config', {})
        
    except Exception as e:
        error_msg = f"Failed to fetch widget config {widget_id}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def get_widgets_by_dashboard_id(dashboard_id: str) -> list[Dict[str, Any]]:
    """Get all widgets for a specific dashboard.
    
    Args:
        dashboard_id: The dashboard ID to fetch widgets for
        
    Returns:
        List of widget dictionaries, ordered by 'order' field or creation date
        
    Raises:
        Exception: If database query fails
    """
    try:
        supabase = get_supabase_client()
        
        logger.info(f"Fetching widgets for dashboard: {dashboard_id}")
        
        # Query widgets for the dashboard, ordered by order field then created_at
        result = supabase.table("widgets").select("*").eq("dashboard_id", dashboard_id).order("order", desc=False).order("created_at", desc=False).execute()
        
        widgets = result.data or []
        logger.info(f"Successfully fetched {len(widgets)} widgets for dashboard: {dashboard_id}")
        
        return widgets
        
    except Exception as e:
        error_msg = f"Failed to fetch widgets for dashboard {dashboard_id}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def get_widget_summary_by_id(widget_id: str) -> Optional[str]:
    """Get a human-readable summary of a widget by ID.
    
    Args:
        widget_id: The widget ID to get summary for
        
    Returns:
        String summary of the widget or None if widget not found
        
    Raises:
        Exception: If database query fails
    """
    try:
        widget = get_widget_by_id(widget_id)
        
        if not widget:
            return None
            
        # Create a human-readable summary
        widget_type = widget.get('type', 'unknown')
        title = widget.get('title', 'Untitled')
        config = widget.get('config', {})
        
        summary = f"Widget '{title}' (Type: {widget_type})"
        
        if widget_type == 'chart':
            chart_type = config.get('chartType', 'unknown chart')
            summary += f" - {chart_type} chart"
            
            # Add data source info if available
            if 'dataSource' in config:
                summary += f" using data source: {config['dataSource']}"
                
        elif widget_type == 'kpi':
            metric = config.get('metric', 'unknown metric')
            summary += f" - displaying {metric}"
            
        elif widget_type == 'table':
            summary += " - data table"
            
        elif widget_type == 'text':
            summary += " - text content block"
            
        # Add creation info
        created_at = widget.get('created_at')
        if created_at:
            summary += f" (Created: {created_at})"
            
        return summary
        
    except Exception as e:
        error_msg = f"Failed to get widget summary {widget_id}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e