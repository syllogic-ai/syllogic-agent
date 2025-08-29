"""
Unit tests for tasks.py functions.
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
)

from actions.tasks import (
    create_task,
    create_tasks_from_delegated_tasks,
    update_task_status,
    get_tasks_by_group,
    get_tasks_by_chat,
    generate_task_group_id,
    format_task_list_message,
)
from agent.models import Task, CreateTaskInput, UpdateTaskInput, DelegatedTask


class TestCreateTask:
    """Test cases for create_task function."""

    def test_create_task_success(self, mock_supabase, sample_task_data):
        """Test successfully creating a task."""
        mock_insert_result = Mock()
        mock_insert_result.data = [sample_task_data]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        # Execute function
        task_input = CreateTaskInput(
            chat_id="chat-123",
            dashboard_id="dashboard-789", 
            task_group_id="group-123",
            title="Test Task",
            description="Test task description",
            status="pending",
            order=1
        )
        result = create_task(mock_supabase, task_input)

        # Verify result
        assert isinstance(result, Task)
        assert result.id == "task-123"
        assert result.title == "Test Task"
        assert result.status == "pending"

    def test_create_task_insert_fails(self, mock_supabase):
        """Test error when insert operation fails."""
        mock_insert_result = Mock()
        mock_insert_result.data = None
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        task_input = CreateTaskInput(
            chat_id="chat-123",
            dashboard_id="dashboard-789",
            task_group_id="group-123", 
            title="Test Task",
            status="pending",
            order=1
        )
        
        with pytest.raises(Exception, match="Failed to create task"):
            create_task(mock_supabase, task_input)

    def test_create_task_database_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("DB Error")

        task_input = CreateTaskInput(
            chat_id="chat-123",
            dashboard_id="dashboard-789",
            task_group_id="group-123",
            title="Test Task", 
            status="pending",
            order=1
        )
        
        with pytest.raises(Exception, match="DB Error"):
            create_task(mock_supabase, task_input)


class TestCreateTasksFromDelegatedTasks:
    """Test cases for create_tasks_from_delegated_tasks function."""

    def test_create_tasks_from_delegated_tasks_success(self, mock_supabase, sample_task_data):
        """Test successfully creating tasks from delegated tasks."""
        # Create delegated task objects
        delegated_task1 = DelegatedTask(
            task_title="Create Chart",
            description="Create a bar chart",
            operation="CREATE",
            widget_type="bar"
        )
        delegated_task2 = DelegatedTask(
            task_title="Update Chart",
            description="Update chart colors",
            operation="UPDATE", 
            widget_type="bar"
        )

        delegated_tasks = [delegated_task1, delegated_task2]

        # Mock insert results for both tasks
        task_data_1 = sample_task_data.copy()
        task_data_1["id"] = "task-123"
        task_data_1["title"] = "Create Chart"
        
        task_data_2 = sample_task_data.copy() 
        task_data_2["id"] = "task-456"
        task_data_2["title"] = "Update Chart"

        # Mock consecutive insert calls
        mock_insert_result_1 = Mock()
        mock_insert_result_1.data = [task_data_1]
        mock_insert_result_2 = Mock()
        mock_insert_result_2.data = [task_data_2]
        
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = [
            mock_insert_result_1, mock_insert_result_2
        ]

        result = create_tasks_from_delegated_tasks(
            mock_supabase,
            delegated_tasks,
            "chat-123", 
            "dashboard-789",
            "group-123"
        )

        assert len(result) == 2
        assert isinstance(result[0], Task)
        assert isinstance(result[1], Task) 
        assert result[0].title == "Create Chart"
        assert result[1].title == "Update Chart"
        
        # Check that delegated tasks were updated
        assert delegated_task1.db_task_id == "task-123"
        assert delegated_task2.db_task_id == "task-456"

    def test_create_tasks_from_delegated_tasks_empty_list(self, mock_supabase):
        """Test creating tasks from empty delegated tasks list."""
        result = create_tasks_from_delegated_tasks(
            mock_supabase, 
            [],
            "chat-123",
            "dashboard-789", 
            "group-123"
        )

        assert isinstance(result, list)
        assert len(result) == 0

    def test_create_tasks_from_delegated_tasks_error(self, mock_supabase):
        """Test handling of database errors."""
        delegated_task = DelegatedTask(
            task_title="Test Task",
            description="Test",
            operation="CREATE",
            widget_type="bar"
        )

        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("DB Error")

        with pytest.raises(Exception, match="DB Error"):
            create_tasks_from_delegated_tasks(
                mock_supabase,
                [delegated_task],
                "chat-123",
                "dashboard-789",
                "group-123"
            )


class TestUpdateTaskStatus:
    """Test cases for update_task_status function."""

    def test_update_task_status_success(self, mock_supabase, sample_task_data):
        """Test successfully updating task status."""
        updated_task_data = sample_task_data.copy()
        updated_task_data["status"] = "completed"
        updated_task_data["completed_at"] = "2023-01-01T11:00:00.000Z"

        mock_update_result = Mock()
        mock_update_result.data = [updated_task_data]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_result

        task_update = UpdateTaskInput(
            task_id="task-123",
            status="completed",
            completed_at="2023-01-01T11:00:00.000Z"
        )
        result = update_task_status(mock_supabase, task_update)

        assert isinstance(result, Task)
        assert result.status == "completed"
        assert result.completed_at == "2023-01-01T11:00:00.000Z"

    def test_update_task_status_fails(self, mock_supabase):
        """Test error when update operation fails."""
        mock_update_result = Mock()
        mock_update_result.data = None
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_result

        task_update = UpdateTaskInput(
            task_id="task-123",
            status="completed"
        )
        
        with pytest.raises(Exception, match="Failed to update task"):
            update_task_status(mock_supabase, task_update)

    def test_update_task_status_database_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception("DB Error")

        task_update = UpdateTaskInput(
            task_id="task-123",
            status="completed"
        )
        
        with pytest.raises(Exception, match="DB Error"):
            update_task_status(mock_supabase, task_update)


class TestGetTasksByGroup:
    """Test cases for get_tasks_by_group function."""

    def test_get_tasks_by_group_success(self, mock_supabase, sample_task_data):
        """Test successfully retrieving tasks by group ID."""
        mock_result = Mock()
        mock_result.data = [sample_task_data]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_tasks_by_group(mock_supabase, "group-123")

        assert len(result) == 1
        assert isinstance(result[0], Task)
        assert result[0].id == "task-123"

    def test_get_tasks_by_group_empty(self, mock_supabase):
        """Test retrieving tasks for non-existent group."""
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_tasks_by_group(mock_supabase, "group-nonexistent")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_tasks_by_group_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("DB Error")

        with pytest.raises(Exception, match="DB Error"):
            get_tasks_by_group(mock_supabase, "group-123")


class TestGetTasksByChat:
    """Test cases for get_tasks_by_chat function."""

    def test_get_tasks_by_chat_success(self, mock_supabase, sample_task_data):
        """Test successfully retrieving tasks by chat ID."""
        mock_result = Mock()
        mock_result.data = [sample_task_data]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_tasks_by_chat(mock_supabase, "chat-123")

        assert len(result) == 1
        assert isinstance(result[0], Task)
        assert result[0].id == "task-123"

    def test_get_tasks_by_chat_empty(self, mock_supabase):
        """Test retrieving tasks for chat with no tasks."""
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_tasks_by_chat(mock_supabase, "chat-123")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_tasks_by_chat_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("DB Error")

        with pytest.raises(Exception, match="DB Error"):
            get_tasks_by_chat(mock_supabase, "chat-123")


class TestGenerateTaskGroupId:
    """Test cases for generate_task_group_id function."""

    def test_generate_task_group_id(self):
        """Test generating task group ID."""
        result = generate_task_group_id("req-123")

        assert isinstance(result, str)
        assert result.startswith("group_req-123_")
        assert len(result) > len("group_req-123_")

    def test_generate_task_group_id_unique(self):
        """Test that generated IDs are unique."""
        id1 = generate_task_group_id("req-123")
        id2 = generate_task_group_id("req-123")

        assert id1 != id2
        assert id1.startswith("group_req-123_")
        assert id2.startswith("group_req-123_")


class TestFormatTaskListMessage:
    """Test cases for format_task_list_message function."""

    def test_format_task_list_message_success(self, sample_task_data):
        """Test formatting task list message."""
        # Create multiple tasks
        task1 = Task(**sample_task_data)
        
        task2_data = sample_task_data.copy()
        task2_data["id"] = "task-456"
        task2_data["title"] = "Second Task"
        task2_data["order"] = 2
        task2_data["status"] = "in-progress"
        task2 = Task(**task2_data)

        tasks = [task1, task2]
        result = format_task_list_message(tasks, "group-123456789")

        assert "📋 **Task List**" in result
        assert "group-123456789"[-8:] in result  # Shows last 8 chars
        assert "1. ⏳ **Test Task** - pending" in result
        assert "2. 🔄 **Second Task** - in-progress" in result

    def test_format_task_list_message_empty(self):
        """Test formatting empty task list."""
        result = format_task_list_message([], "group-123")

        assert result == "No tasks found."

    def test_format_task_list_message_with_descriptions(self, sample_task_data):
        """Test formatting task list with descriptions."""
        task = Task(**sample_task_data)
        tasks = [task]
        
        result = format_task_list_message(tasks, "group-123")

        assert "Test task description" in result
        assert "_Test task description_" in result  # Italic formatting

    def test_format_task_list_message_different_statuses(self, sample_task_data):
        """Test formatting with different task statuses."""
        statuses = ["pending", "in-progress", "completed", "failed"]
        emojis = ["⏳", "🔄", "✅", "❌"]
        
        tasks = []
        for i, status in enumerate(statuses):
            task_data = sample_task_data.copy()
            task_data["id"] = f"task-{i}"
            task_data["status"] = status
            task_data["order"] = i + 1
            tasks.append(Task(**task_data))

        result = format_task_list_message(tasks, "group-123")

        for i, emoji in enumerate(emojis):
            assert f"{i+1}. {emoji}" in result