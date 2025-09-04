"""Data reading tools for the Top Level Supervisor.

This module provides wrapper functions around actions.dashboard helper functions.
Following CLAUDE.md guidelines, actual helper logic is in src/actions/.
"""

import logging
from typing import Dict

from actions.utils import import_actions_dashboard

logger = logging.getLogger(__name__)


def get_available_data(dashboard_id: str) -> Dict[str, any]:
    """Get available data files and their schemas for a dashboard.
    
    This is a wrapper around the helper function in actions.dashboard.
    
    Args:
        dashboard_id: Dashboard identifier
        
    Returns:
        Dict containing available files, schemas, and summary
    """
    try:
        # Import dashboard helper functions using robust import
        dashboard_module = import_actions_dashboard()
        
        # Use the helper function from actions.dashboard
        return dashboard_module.get_available_data(dashboard_id)
        
    except Exception as e:
        logger.error(f"Error in data reader wrapper: {e}")
        return {
            "available_files": [],
            "file_schemas": [],
            "data_summary": f"Error retrieving data: {str(e)}"
        }