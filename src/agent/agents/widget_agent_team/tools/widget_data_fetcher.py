"""Tool for fetching widget data from the database for reference purposes."""

from typing import Any, Dict, Optional

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Handle imports for different execution contexts
try:
    from actions.dashboard import get_widget_from_widget_id
except ImportError:
    import sys
    import os
    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.dashboard import get_widget_from_widget_id


def fetch_reference_widget_config(reference_widget_id: str) -> Dict[str, Any]:
    """Fetch config data from reference widget for text block creation.
    
    Args:
        reference_widget_id: Widget ID to fetch configuration from
        
    Returns:
        Dict containing widget configuration data, or empty dict if not found
        
    Raises:
        Exception: If widget fetch fails
    """
    try:
        logger.info(f"Fetching reference widget config for widget_id: {reference_widget_id}")
        
        widget = get_widget_from_widget_id(reference_widget_id)
        if not widget:
            logger.warning(f"Reference widget {reference_widget_id} not found")
            return {}
            
        # Return the config data which contains chart configuration
        config = widget.config or {}
        
        logger.info(f"Successfully fetched config for reference widget {reference_widget_id}")
        return config
        
    except Exception as e:
        logger.error(f"Error fetching reference widget {reference_widget_id}: {str(e)}")
        raise


def fetch_reference_widget_full_data(reference_widget_id: str) -> Optional[Dict[str, Any]]:
    """Fetch complete widget data including title, description, config, and data.
    
    Args:
        reference_widget_id: Widget ID to fetch complete data from
        
    Returns:
        Dict containing complete widget information, or None if not found
        
    Raises:
        Exception: If widget fetch fails
    """
    try:
        logger.info(f"Fetching complete reference widget data for widget_id: {reference_widget_id}")
        
        widget = get_widget_from_widget_id(reference_widget_id)
        if not widget:
            logger.warning(f"Reference widget {reference_widget_id} not found")
            return None
            
        # Return all relevant widget data for text block creation
        widget_data = {
            "widget_id": widget.id,
            "title": widget.title,
            "type": widget.type,
            "config": widget.config or {},
            "data": widget.data or {},
            "dashboard_id": widget.dashboard_id,
            "created_at": widget.created_at,
            "updated_at": widget.updated_at,
            # Note: description field not available in database schema
        }
        
        logger.info(f"Successfully fetched complete data for reference widget {reference_widget_id}")
        return widget_data
        
    except Exception as e:
        logger.error(f"Error fetching complete reference widget data {reference_widget_id}: {str(e)}")
        raise