"""Data fetching tool for widget processing."""

import os
import sys
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agent.models import (
    ColumnInfo,
    FileSampleData,
    FileSchema,
    WidgetAgentState,
)
from actions.utils import import_actions_dashboard


@tool
def fetch_data_tool(
    state: Annotated[WidgetAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Fetches data from file IDs in state and updates state with fetched data."""
    try:
        # Import file fetching functions from actions using robust import
        try:
            dashboard_module = import_actions_dashboard()
            get_data_from_file = dashboard_module.get_data_from_file
            get_sample_from_file = dashboard_module.get_sample_from_file
            get_schema_from_file = dashboard_module.get_schema_from_file
        except ImportError as e:
            error_msg = f"Failed to import actions.dashboard module: {str(e)}"
            return Command(
                update={
                    "error_messages": state.error_messages + [error_msg],
                    "messages": [
                        ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                    ],
                }
            )

        # Extract file_ids from state
        file_ids = state.file_ids
        if not file_ids:
            return Command(
                update={
                    "error_messages": state.error_messages
                    + ["No file IDs provided for data fetching"],
                    "messages": [
                        ToolMessage(
                            content="Error: No file IDs provided for data fetching",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        raw_data = {}
        schemas_info = []
        samples_info = []

        for file_id in file_ids:
            try:
                # Fetch raw data for processing
                file_data = get_data_from_file(file_id)
                raw_data[file_id] = file_data

                # Always fetch schema and sample data
                schema_data = get_schema_from_file(file_id)
                schemas_info.append(
                    {
                        "file_id": file_id,
                        "columns": len(schema_data.get("columns", [])),
                        "total_rows": schema_data.get("total_rows", 0),
                        "column_names": [
                            col["name"] for col in schema_data.get("columns", [])
                        ],
                    }
                )

                sample_data = get_sample_from_file(file_id)
                samples_info.append(
                    {
                        "file_id": file_id,
                        "headers": sample_data.get("headers", []),
                        "sample_rows": sample_data.get("sample_rows_returned", 0),
                    }
                )

            except Exception as file_error:
                return f"Failed to fetch data for file {file_id}: {str(file_error)}"

        # Convert raw pandas DataFrames to JSON for state storage
        raw_data_json = {}
        for file_id, df in raw_data.items():
            # Convert DataFrame to dict with records orientation
            raw_data_json[file_id] = df.to_dict("records")

        # Create FileSchema objects
        file_schemas_objects = [
            FileSchema(
                file_id=info["file_id"],
                columns=[
                    ColumnInfo(name=col, type="unknown", null_count=0, unique_count=0)
                    for col in info["column_names"]
                ],
                total_rows=info["total_rows"],
                total_columns=len(info["column_names"]),
            )
            for info in schemas_info
        ]

        # Create FileSampleData objects
        file_sample_objects = [
            FileSampleData(
                file_id=info["file_id"],
                headers=info["headers"],
                rows=[],  # Could populate from sample data if needed
                total_rows_in_file=schemas_info[i]["total_rows"]
                if i < len(schemas_info)
                else 0,
                sample_rows_returned=info["sample_rows"],
            )
            for i, info in enumerate(samples_info)
        ]

        success_message = (
            f"Successfully fetched data from {len(file_ids)} files:\n"
            + "\n".join(
                [
                    f"File {info['file_id']}: {info['total_rows']} rows, columns: {', '.join(info['column_names'][:5])}{'...' if len(info['column_names']) > 5 else ''}"
                    for info in schemas_info
                ]
            )
        )

        # Return Command to immediately update state
        return Command(
            update={
                "raw_file_data": raw_data_json,
                "file_schemas": file_schemas_objects,
                "file_sample_data": file_sample_objects,
                "messages": [
                    ToolMessage(content=success_message, tool_call_id=tool_call_id)
                ],
            }
        )

    except Exception as e:
        error_msg = f"Data fetch error: {str(e)}"
        return Command(
            update={
                "error_messages": state.error_messages + [error_msg],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
            }
        )