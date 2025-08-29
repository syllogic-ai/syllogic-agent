"""
Unit tests for messages.py functions.
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
)

from actions.messages import (
    create_message,
    create_task_list_message,
    get_messages_by_chat,
    get_messages_by_task_group,
    append_chat_message,
)
from agent.models import Message, CreateMessageInput


class TestCreateMessage:
    """Test cases for create_message function."""

    def test_create_message_success(self, mock_supabase, sample_message_data):
        """Test successfully creating a message."""
        # Setup table mocks properly
        mock_messages_table = Mock()
        mock_chats_table = Mock()
        
        def table_side_effect(table_name):
            if table_name == "messages":
                return mock_messages_table
            elif table_name == "chats":
                return mock_chats_table
            return Mock()
        
        mock_supabase.table.side_effect = table_side_effect

        # Mock the insert chain for messages
        mock_insert_result = Mock()
        mock_insert_result.data = [sample_message_data]
        mock_messages_table.insert.return_value.execute.return_value = mock_insert_result

        # Mock the chat metadata update (count query and update)
        mock_count_result = Mock()
        mock_count_result.count = 1
        mock_messages_table.select.return_value.eq.return_value.execute.return_value = mock_count_result
        mock_chats_table.update.return_value.eq.return_value.execute.return_value = Mock()

        # Execute function
        message_input = CreateMessageInput(
            chat_id="chat-123", 
            role="user", 
            content="Test message"
        )
        result = create_message(mock_supabase, message_input)

        # Verify result
        assert isinstance(result, Message)
        assert result.id == "message-123"
        assert result.content == "New message"
        assert result.role == "user"

    def test_create_message_with_task_group(self, mock_supabase, sample_message_data):
        """Test creating a message with task group ID."""
        # Update sample data for this test
        task_message_data = sample_message_data.copy()
        task_message_data["task_group_id"] = "group-456"
        task_message_data["message_type"] = "task-list"

        mock_insert_result = Mock()
        mock_insert_result.data = [task_message_data]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        # Mock the chat metadata update
        mock_count_result = Mock()
        mock_count_result.count = 1
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()

        message_input = CreateMessageInput(
            chat_id="chat-123",
            role="ai", 
            content="Task list content",
            message_type="task-list",
            task_group_id="group-456"
        )
        result = create_message(mock_supabase, message_input)

        assert isinstance(result, Message)
        assert result.task_group_id == "group-456"
        assert result.message_type == "task-list"

    def test_create_message_insert_fails(self, mock_supabase):
        """Test error when insert operation fails."""
        mock_insert_result = Mock()
        mock_insert_result.data = None
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        message_input = CreateMessageInput(
            chat_id="chat-123", role="user", content="Test message"
        )
        
        with pytest.raises(Exception, match="Failed to create message"):
            create_message(mock_supabase, message_input)

    def test_create_message_database_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("DB Error")

        message_input = CreateMessageInput(
            chat_id="chat-123", role="user", content="Test message"
        )
        
        with pytest.raises(Exception, match="DB Error"):
            create_message(mock_supabase, message_input)


class TestCreateTaskListMessage:
    """Test cases for create_task_list_message function."""

    def test_create_task_list_success(self, mock_supabase, sample_message_data):
        """Test successfully creating a task list message."""
        # Update sample data for task list message
        task_list_data = sample_message_data.copy()
        task_list_data["role"] = "ai"
        task_list_data["message_type"] = "task-list"
        task_list_data["task_group_id"] = "group-456"
        task_list_data["content"] = "Task list content"

        mock_insert_result = Mock()
        mock_insert_result.data = [task_list_data]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        # Mock the chat metadata update
        mock_count_result = Mock()
        mock_count_result.count = 1
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()

        result = create_task_list_message(
            mock_supabase, 
            "chat-123", 
            "group-456", 
            "Task list content"
        )

        assert isinstance(result, Message)
        assert result.role == "ai"
        assert result.message_type == "task-list"
        assert result.task_group_id == "group-456"


class TestGetMessagesByChat:
    """Test cases for get_messages_by_chat function."""

    def test_get_messages_success(self, mock_supabase, sample_message_data):
        """Test successfully retrieving messages by chat ID."""
        mock_result = Mock()
        mock_result.data = [sample_message_data]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_messages_by_chat(mock_supabase, "chat-123")

        assert len(result) == 1
        assert isinstance(result[0], Message)
        assert result[0].id == "message-123"

    def test_get_messages_with_limit(self, mock_supabase, sample_message_data):
        """Test retrieving messages with limit."""
        mock_result = Mock()
        mock_result.data = [sample_message_data]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        result = get_messages_by_chat(mock_supabase, "chat-123", limit=5)

        assert len(result) == 1
        assert isinstance(result[0], Message)

    def test_get_messages_empty_result(self, mock_supabase):
        """Test retrieving messages when none exist."""
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_messages_by_chat(mock_supabase, "chat-123")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_messages_database_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("DB Error")

        with pytest.raises(Exception, match="DB Error"):
            get_messages_by_chat(mock_supabase, "chat-123")


class TestGetMessagesByTaskGroup:
    """Test cases for get_messages_by_task_group function."""

    def test_get_messages_by_task_group_success(self, mock_supabase, sample_message_data):
        """Test successfully retrieving messages by task group ID."""
        # Update sample data for task group
        task_group_data = sample_message_data.copy()
        task_group_data["task_group_id"] = "group-456"

        mock_result = Mock()
        mock_result.data = [task_group_data]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_messages_by_task_group(mock_supabase, "group-456")

        assert len(result) == 1
        assert isinstance(result[0], Message)
        assert result[0].task_group_id == "group-456"

    def test_get_messages_by_task_group_empty(self, mock_supabase):
        """Test retrieving messages for non-existent task group."""
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_result

        result = get_messages_by_task_group(mock_supabase, "group-nonexistent")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_messages_by_task_group_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("DB Error")

        with pytest.raises(Exception, match="DB Error"):
            get_messages_by_task_group(mock_supabase, "group-456")


class TestAppendChatMessage:
    """Test cases for append_chat_message function (backward compatibility)."""

    def test_append_chat_message_success(self, mock_supabase, sample_message_data):
        """Test append_chat_message as wrapper for create_message."""
        mock_insert_result = Mock()
        mock_insert_result.data = [sample_message_data]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        # Mock the chat metadata update
        mock_count_result = Mock()
        mock_count_result.count = 1
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()

        message_input = CreateMessageInput(
            chat_id="chat-123", role="user", content="Test message"
        )
        result = append_chat_message(mock_supabase, message_input)

        assert isinstance(result, Message)
        assert result.id == "message-123"