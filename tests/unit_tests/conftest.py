"""
Test configuration and fixtures for backend tests.
"""

from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.fixture
def mock_supabase():
    """Mock Supabase client for testing."""
    mock_client = Mock()

    # Mock table operations
    mock_table = Mock()
    mock_client.table.return_value = mock_table

    # Mock query operations
    mock_query = Mock()
    mock_table.select.return_value = mock_query
    mock_table.insert.return_value = mock_query
    mock_table.update.return_value = mock_query
    mock_table.delete.return_value = mock_query

    # Mock query chaining
    mock_query.eq.return_value = mock_query
    mock_query.single.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.lt.return_value = mock_query
    mock_query.in_.return_value = mock_query

    return mock_client


@pytest.fixture
def sample_chat_data():
    """Sample chat data for testing."""
    return {
        "id": "chat-123",
        "user_id": "user-456",
        "dashboard_id": "dashboard-789",
        "title": "Test Chat",
        "conversation": [
            {
                "role": "user",
                "message": "Hello",
                "timestamp": "2023-01-01T10:00:00.000Z",
            },
            {
                "role": "system",
                "message": "Hi there!",
                "timestamp": "2023-01-01T10:00:01.000Z",
            },
        ],
        "created_at": "2023-01-01T09:00:00.000Z",
        "updated_at": "2023-01-01T10:00:00.000Z",
    }


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "id": "job-123",
        "user_id": "user-456",
        "dashboard_id": "dashboard-789",
        "job_type": "widget_creation",
        "status": "pending",
        "progress": 0,
        "error": None,
        "created_at": "2023-01-01T09:00:00.000Z",
        "updated_at": "2023-01-01T09:00:00.000Z",
        "started_at": None,
        "completed_at": None,
        "processing_time_ms": None,
        "queue_time_ms": None,
    }


@pytest.fixture
def sample_widget_data():
    """Sample widget data for testing."""
    return {
        "id": "widget-123",
        "dashboard_id": "dashboard-789",
        "title": "Test Widget",
        "type": "chart",
        "config": {"chart_type": "bar", "x_axis": "date"},
        "data": {"chart_data": [{"x": 1, "y": 2}]},
        "sql": "SELECT * FROM table",
        "layout": {"x": 0, "y": 0, "w": 4, "h": 3},
        "chat_id": "chat-123",
        "order": 1,
        "is_configured": True,
        "cache_key": "cache-key-123",
        "last_data_fetch": "2023-01-01T10:00:00.000Z",
        "created_at": "2023-01-01T09:00:00.000Z",
        "updated_at": "2023-01-01T10:00:00.000Z",
    }


@pytest.fixture
def sample_file_data():
    """Sample file data for testing."""
    return {
        "id": "file-123",
        "user_id": "user-456",
        "dashboard_id": "dashboard-789",
        "file_type": "original",
        "original_filename": "test.csv",
        "sanitized_filename": "test-sanitized.csv",
        "storage_path": "uploads/test-sanitized.csv",
        "mime_type": "text/csv",
        "size": 1024,
        "status": "ready",
        "created_at": "2023-01-01T09:00:00.000Z",
    }


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value": [100, 150, 200],
            "category": ["A", "B", "A"],
            "percentage": [10.5, 15.2, 20.8],
        }
    )


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """date,value,category,percentage
2023-01-01,100,A,10.5
2023-01-02,150,B,15.2
2023-01-03,200,A,20.8"""


class MockResponse:
    """Mock HTTP response for testing file downloads."""

    def __init__(self, text_content: str, status_code: int = 200):
        self.text = text_content
        self.content = text_content.encode()
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for testing file downloads."""

    def _mock_get(url, **kwargs):
        if "test.csv" in url:
            csv_content = """date,value,category,percentage
2023-01-01,100,A,10.5
2023-01-02,150,B,15.2
2023-01-03,200,A,20.8"""
            return MockResponse(csv_content)
        else:
            return MockResponse("", 404)

    return _mock_get
