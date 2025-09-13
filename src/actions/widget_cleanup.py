"""Widget cleanup utilities for removing unconfigured widgets.

This module provides functions to clean up widgets that have is_configured=False
from the Supabase widget table. This is typically run at the end of agentic flow
execution to maintain database hygiene.
"""

from typing import Dict, Any, List

from config import get_supabase_client

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def cleanup_unconfigured_widgets() -> Dict[str, Any]:
    """Clean up all widgets with is_configured=False from the database.
    
    This function removes all widgets that have not been properly configured,
    which helps maintain database hygiene and prevents accumulation of incomplete
    widget records.
    
    Returns:
        Dict containing cleanup results:
        - success: Boolean indicating if cleanup was successful
        - deleted_count: Number of widgets deleted
        - deleted_widgets: List of deleted widget IDs
        - error: Error message if cleanup failed
        
    Raises:
        Exception: If database operation fails
    """
    try:
        supabase = get_supabase_client()
        
        # First, get all unconfigured widgets to log what we're deleting
        unconfigured_widgets_result = supabase.table("widgets").select("id, title, type, dashboard_id").eq("is_configured", False).execute()
        
        if not unconfigured_widgets_result.data:
            logger.info("No unconfigured widgets found for cleanup")
            return {
                "success": True,
                "deleted_count": 0,
                "deleted_widgets": [],
                "message": "No unconfigured widgets found"
            }
        
        unconfigured_widgets = unconfigured_widgets_result.data
        widget_ids = [widget["id"] for widget in unconfigured_widgets]
        
        logger.info(f"Found {len(unconfigured_widgets)} unconfigured widgets to clean up")
        
        # Log details of widgets being deleted for debugging
        for widget in unconfigured_widgets:
            logger.debug(f"Cleaning up widget: {widget['id']} - {widget['title']} ({widget['type']}) in dashboard {widget['dashboard_id']}")
        
        # Delete all unconfigured widgets
        delete_result = supabase.table("widgets").delete().eq("is_configured", False).execute()
        
        if delete_result.data is not None:
            deleted_count = len(delete_result.data)
            logger.info(f"Successfully cleaned up {deleted_count} unconfigured widgets")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "deleted_widgets": widget_ids,
                "message": f"Successfully cleaned up {deleted_count} unconfigured widgets"
            }
        else:
            # This shouldn't happen, but handle gracefully
            logger.warning("Delete operation returned no data - widgets may have been deleted")
            return {
                "success": True,
                "deleted_count": len(widget_ids),
                "deleted_widgets": widget_ids,
                "message": f"Cleaned up {len(widget_ids)} unconfigured widgets (count may be approximate)"
            }
            
    except Exception as e:
        error_msg = f"Failed to cleanup unconfigured widgets: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "deleted_count": 0,
            "deleted_widgets": [],
            "error": error_msg
        }


def cleanup_unconfigured_widgets_by_dashboard(dashboard_id: str) -> Dict[str, Any]:
    """Clean up unconfigured widgets for a specific dashboard.
    
    Args:
        dashboard_id: The dashboard ID to clean up widgets for
        
    Returns:
        Dict containing cleanup results:
        - success: Boolean indicating if cleanup was successful
        - deleted_count: Number of widgets deleted
        - deleted_widgets: List of deleted widget IDs
        - error: Error message if cleanup failed
    """
    try:
        supabase = get_supabase_client()
        
        # Get unconfigured widgets for the specific dashboard
        unconfigured_widgets_result = supabase.table("widgets").select("id, title, type").eq("is_configured", False).eq("dashboard_id", dashboard_id).execute()
        
        if not unconfigured_widgets_result.data:
            logger.info(f"No unconfigured widgets found for dashboard {dashboard_id}")
            return {
                "success": True,
                "deleted_count": 0,
                "deleted_widgets": [],
                "message": f"No unconfigured widgets found for dashboard {dashboard_id}"
            }
        
        unconfigured_widgets = unconfigured_widgets_result.data
        widget_ids = [widget["id"] for widget in unconfigured_widgets]
        
        logger.info(f"Found {len(unconfigured_widgets)} unconfigured widgets to clean up for dashboard {dashboard_id}")
        
        # Log details of widgets being deleted
        for widget in unconfigured_widgets:
            logger.debug(f"Cleaning up widget: {widget['id']} - {widget['title']} ({widget['type']})")
        
        # Delete unconfigured widgets for this dashboard
        delete_result = supabase.table("widgets").delete().eq("is_configured", False).eq("dashboard_id", dashboard_id).execute()
        
        if delete_result.data is not None:
            deleted_count = len(delete_result.data)
            logger.info(f"Successfully cleaned up {deleted_count} unconfigured widgets for dashboard {dashboard_id}")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "deleted_widgets": widget_ids,
                "message": f"Successfully cleaned up {deleted_count} unconfigured widgets for dashboard {dashboard_id}"
            }
        else:
            logger.warning("Delete operation returned no data - widgets may have been deleted")
            return {
                "success": True,
                "deleted_count": len(widget_ids),
                "deleted_widgets": widget_ids,
                "message": f"Cleaned up {len(widget_ids)} unconfigured widgets for dashboard {dashboard_id} (count may be approximate)"
            }
            
    except Exception as e:
        error_msg = f"Failed to cleanup unconfigured widgets for dashboard {dashboard_id}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "deleted_count": 0,
            "deleted_widgets": [],
            "error": error_msg
        }


def get_unconfigured_widgets_count() -> Dict[str, Any]:
    """Get the count of unconfigured widgets without deleting them.
    
    Returns:
        Dict containing count information:
        - count: Number of unconfigured widgets
        - widgets: List of unconfigured widget details
        - error: Error message if query failed
    """
    try:
        supabase = get_supabase_client()
        
        # Get unconfigured widgets with basic info
        result = supabase.table("widgets").select("id, title, type, dashboard_id, created_at").eq("is_configured", False).execute()
        
        if result.data is not None:
            return {
                "count": len(result.data),
                "widgets": result.data,
                "message": f"Found {len(result.data)} unconfigured widgets"
            }
        else:
            return {
                "count": 0,
                "widgets": [],
                "message": "No unconfigured widgets found"
            }
            
    except Exception as e:
        error_msg = f"Failed to get unconfigured widgets count: {str(e)}"
        logger.error(error_msg)
        return {
            "count": 0,
            "widgets": [],
            "error": error_msg
        }
