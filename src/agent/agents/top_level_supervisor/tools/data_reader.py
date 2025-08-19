"""Data reading tools for the Top Level Supervisor."""

import logging
import os
import sys
from typing import Dict, List

from actions.utils import import_config, import_actions_dashboard

logger = logging.getLogger(__name__)


def get_available_data(dashboard_id: str) -> Dict[str, any]:
    """Get available data files and their schemas for a dashboard.
    
    Args:
        dashboard_id: Dashboard identifier
        
    Returns:
        Dict containing available files, schemas, and summary
    """
    try:
        # Import config and dashboard functions using robust import
        config_module = import_config()
        actions_dashboard = import_actions_dashboard()
        
        supabase = config_module.get_supabase_client()
        
        # Get all files associated with this dashboard
        response = (
            supabase.table("files")
            .select("*")
            .eq("dashboard_id", dashboard_id)
            .execute()
        )
        
        if not response.data:
            return {
                "available_files": [],
                "file_schemas": [],
                "data_summary": "No data files available for this dashboard."
            }
        
        available_files = []
        file_schemas = []
        
        for file_record in response.data:
            file_id = file_record["id"]
            file_name = file_record.get("name", f"File {file_id}")
            available_files.append(file_id)
            
            try:
                # Get schema for this file
                schema = actions_dashboard.get_schema_from_file(file_id)
                file_schemas.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "schema": schema,
                    "file_type": file_record.get("type", "unknown")
                })
            except Exception as e:
                logger.warning(f"Could not get schema for file {file_id}: {e}")
                file_schemas.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "schema": None,
                    "error": str(e)
                })
        
        # Create summary of available data
        data_summary = _create_data_summary(file_schemas)
        
        return {
            "available_files": available_files,
            "file_schemas": file_schemas,
            "data_summary": data_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting available data: {e}")
        return {
            "available_files": [],
            "file_schemas": [],
            "data_summary": f"Error retrieving data: {str(e)}"
        }


def _create_data_summary(file_schemas: List[Dict]) -> str:
    """Create a human-readable summary of available data."""
    if not file_schemas:
        return "No data files available."
    
    summary_parts = []
    summary_parts.append(f"Available data files: {len(file_schemas)} file(s)")
    
    for file_info in file_schemas:
        file_name = file_info.get("file_name", "Unknown")
        file_id = file_info.get("file_id", "Unknown")
        
        if file_info.get("error"):
            summary_parts.append(f"- {file_name} (ID: {file_id}): Error loading schema")
            continue
            
        schema = file_info.get("schema")
        if not schema:
            summary_parts.append(f"- {file_name} (ID: {file_id}): No schema available")
            continue
            
        columns = schema.get("columns", [])
        total_rows = schema.get("total_rows", 0)
        
        column_summary = []
        for col in columns[:5]:  # Show first 5 columns
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "unknown")
            column_summary.append(f"{col_name} ({col_type})")
        
        if len(columns) > 5:
            column_summary.append(f"... and {len(columns) - 5} more columns")
        
        summary_parts.append(
            f"- {file_name} (ID: {file_id}): {total_rows} rows, "
            f"columns: {', '.join(column_summary)}"
        )
    
    return "\n".join(summary_parts)