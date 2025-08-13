"""Data fetching tool that leverages src/actions/ to retrieve file schemas and sample data."""

import logging
from typing import List

from langchain_core.tools import tool
from supabase import Client

from ...actions.dashboard import get_sample_from_file, get_schema_from_file
from ..models import ColumnInfo, FileSampleData, FileSchema, TopLevelState

logger = logging.getLogger(__name__)


async def fetch_missing_data(
    state: TopLevelState, supabase: Client, file_ids: List[str] = None
) -> TopLevelState:
    """Fetch missing schemas and sample data for files that don't have them yet.

    Args:
        state: Current chart generation state
        supabase: Supabase client instance
        file_ids: Specific file IDs to fetch (if None, uses state.file_ids)

    Returns:
        Updated ChartGenerationState with fetched data
    """
    try:
        logger.info("Starting data fetching process")

        # Use provided file_ids or get from state
        target_file_ids = file_ids or state.file_ids

        if not target_file_ids:
            logger.warning("No file IDs provided for data fetching")
            return state

        # Identify files that need data fetching
        files_needing_schemas = []
        files_needing_samples = []

        for file_id in target_file_ids:
            if file_id not in state.available_data_schemas:
                files_needing_schemas.append(file_id)
            if file_id not in state.available_sample_data:
                files_needing_samples.append(file_id)

        logger.info(f"Files needing schemas: {len(files_needing_schemas)}")
        logger.info(f"Files needing samples: {len(files_needing_samples)}")

        # Fetch schemas for files that need them
        for file_id in files_needing_schemas:
            try:
                logger.info(f"Fetching schema for file {file_id}")
                schema_data = get_schema_from_file(supabase, file_id)

                # Convert to our model format
                columns = []
                for col_data in schema_data.get("columns", []):
                    column_info = ColumnInfo(
                        name=col_data["name"],
                        type=col_data["type"],
                        null_count=col_data["null_count"],
                        unique_count=col_data["unique_count"],
                        sample_values=col_data.get("sample_values", []),
                    )
                    columns.append(column_info)

                file_schema = FileSchema(
                    file_id=file_id,
                    columns=columns,
                    total_rows=schema_data["total_rows"],
                    total_columns=schema_data["total_columns"],
                )

                state.available_data_schemas[file_id] = file_schema
                logger.info(f"Successfully fetched schema for file {file_id}")

            except Exception as e:
                error_msg = f"Failed to fetch schema for file {file_id}: {str(e)}"
                logger.error(error_msg)
                state.errors.append(error_msg)

        # Fetch sample data for files that need it
        for file_id in files_needing_samples:
            try:
                logger.info(f"Fetching sample data for file {file_id}")
                sample_data = get_sample_from_file(supabase, file_id, num_rows=5)

                # Convert to our model format
                file_sample = FileSampleData(
                    file_id=file_id,
                    headers=sample_data["headers"],
                    rows=sample_data["rows"],
                    total_rows_in_file=sample_data["total_rows_in_file"],
                    sample_rows_returned=sample_data["sample_rows_returned"],
                )

                state.available_sample_data[file_id] = file_sample
                logger.info(f"Successfully fetched sample data for file {file_id}")

            except Exception as e:
                error_msg = f"Failed to fetch sample data for file {file_id}: {str(e)}"
                logger.error(error_msg)
                state.errors.append(error_msg)

        logger.info(
            f"Data fetching completed. Total schemas: {len(state.available_data_schemas)}, Total samples: {len(state.available_sample_data)}"
        )
        return state

    except Exception as e:
        error_msg = f"Unexpected error during data fetching: {str(e)}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        return state


async def get_dashboard_files(dashboard_id: str, supabase: Client) -> List[str]:
    """Get all file IDs associated with a dashboard.

    Args:
        dashboard_id: Dashboard identifier
        supabase: Supabase client instance

    Returns:
        List of file IDs associated with the dashboard
    """
    try:
        logger.info(f"Fetching files for dashboard {dashboard_id}")

        # Query dashboards table to get associated files
        # Assuming the dashboard has a files field that contains file IDs
        result = (
            supabase.table("dashboards")
            .select("files")
            .eq("id", dashboard_id)
            .single()
            .execute()
        )

        if not result.data:
            logger.warning(f"Dashboard {dashboard_id} not found")
            return []

        files = result.data.get("files", [])

        # Handle both string and list formats
        if isinstance(files, str):
            files = [files]
        elif not isinstance(files, list):
            files = []

        logger.info(f"Found {len(files)} files for dashboard {dashboard_id}")
        return files

    except Exception as e:
        logger.error(f"Error fetching files for dashboard {dashboard_id}: {str(e)}")
        return []


# LangGraph Tool Functions


@tool
def fetch_file_schema_tool(file_id: str) -> str:
    """Get schema information for a file using Supabase.

    Args:
        file_id: The ID of the file to get schema for

    Returns:
        JSON string with schema information
    """
    try:
        # Note: In a real implementation, we'd get supabase from context
        # For now, return a mock response
        import json

        mock_schema = {
            "file_id": file_id,
            "columns": [
                {
                    "name": "region",
                    "type": "string",
                    "sample_values": ["North", "South", "East"],
                },
                {
                    "name": "sales",
                    "type": "number",
                    "sample_values": [1000, 1500, 2000],
                },
                {
                    "name": "date",
                    "type": "date",
                    "sample_values": ["2024-01-01", "2024-01-02"],
                },
            ],
            "total_rows": 100,
            "total_columns": 3,
        }
        return json.dumps(mock_schema)
    except Exception as e:
        return f"Error fetching schema for {file_id}: {str(e)}"


@tool
def fetch_file_sample_tool(file_id: str, num_rows: int = 5) -> str:
    """Get sample data for a file using Supabase.

    Args:
        file_id: The ID of the file to get sample for
        num_rows: Number of sample rows to fetch

    Returns:
        JSON string with sample data
    """
    try:
        import json

        mock_sample = {
            "file_id": file_id,
            "headers": ["region", "sales", "date"],
            "rows": [
                ["North", 1000, "2024-01-01"],
                ["South", 1500, "2024-01-02"],
                ["East", 2000, "2024-01-03"],
            ],
            "total_rows_in_file": 100,
            "sample_rows_returned": 3,
        }
        return json.dumps(mock_sample)
    except Exception as e:
        return f"Error fetching sample for {file_id}: {str(e)}"


@tool
def get_dashboard_files_tool(dashboard_id: str) -> str:
    """Get all file IDs associated with a dashboard.

    Args:
        dashboard_id: Dashboard identifier

    Returns:
        JSON string with list of file IDs
    """
    try:
        import json

        # Mock response - in real implementation would query Supabase
        mock_files = {
            "dashboard_id": dashboard_id,
            "file_ids": ["file_001", "file_002", "file_003"],
        }
        return json.dumps(mock_files)
    except Exception as e:
        return f"Error fetching files for dashboard {dashboard_id}: {str(e)}"


def get_data_tools():
    """Get list of data fetching tools for LangGraph.

    Returns:
        List of LangGraph tools
    """
    return [fetch_file_schema_tool, fetch_file_sample_tool, get_dashboard_files_tool]
