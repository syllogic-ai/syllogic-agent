"""Tool to fetch widget details by widget_id.

Following CLAUDE.md guidelines, this agent tool wraps helper functions from src/actions/.
All database logic is in actions.dashboard helper functions.
"""

import json
import logging
from typing import Dict, Any, List
from langchain_core.tools import tool

from actions.dashboard import fetch_reference_widget_details, fetch_multiple_reference_widget_details

logger = logging.getLogger(__name__)


@tool
def fetch_reference_widgets_details(
    reference_widget_id: str = None,
    reference_widget_ids: List[str] = None
) -> str:
    """Fetch reference widget details from database using widget IDs.
    
    This agent tool wraps helper functions from actions.dashboard following CLAUDE.md guidelines.
    Fetches widget details that text blocks need to reference or explain.
    
    Args:
        reference_widget_id: Single widget ID to fetch (optional)
        reference_widget_ids: List of widget IDs to fetch (optional)
    
    Returns:
        JSON string containing fetched widget details
    """
    try:
        reference_widgets = []
        
        # Check for single reference widget ID
        if reference_widget_id:
            try:
                widget_details = fetch_reference_widget_details(reference_widget_id)
                reference_widgets.append(widget_details)
                logger.info(f"Fetched details for single reference widget: {reference_widget_id}")
            except Exception as e:
                error_msg = f"Failed to fetch reference widget {reference_widget_id}: {str(e)}"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
        
        # Check for multiple reference widget IDs  
        if reference_widget_ids:
            try:
                multiple_widgets = fetch_multiple_reference_widget_details(reference_widget_ids)
                reference_widgets.extend(multiple_widgets)
                logger.info(f"Fetched details for {len(multiple_widgets)} reference widgets from IDs: {reference_widget_ids}")
            except Exception as e:
                error_msg = f"Failed to fetch reference widgets {reference_widget_ids}: {str(e)}"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
        
        # If no reference widgets specified, return empty result
        if not reference_widgets:
            logger.info("No reference widget IDs provided")
            return "No reference widget IDs provided - text block will be created without widget references."
        
        # Format the results for LLM consumption
        reference_data_json = json.dumps(reference_widgets, indent=2, default=str)
        
        success_msg = f"Successfully fetched {len(reference_widgets)} reference widget(s) from database:\n"
        for widget in reference_widgets:
            success_msg += f"- {widget['title']} ({widget['widget_type']}, ID: {widget['widget_id']})\n"
        
        success_msg += f"\nWidget details (JSON):\n{reference_data_json}"
        
        return success_msg
        
    except Exception as e:
        error_msg = f"Unexpected error in fetch_reference_widgets_details: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"