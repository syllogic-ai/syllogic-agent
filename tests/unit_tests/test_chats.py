"""
Unit tests for backward compatibility of chat functions.
These tests ensure that the new message-based system maintains compatibility with old chat interfaces.
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
)

from actions.messages import append_chat_message
from agent.models import Message, CreateMessageInput as ChatMessageInput


class TestBackwardCompatibility:
    """Test cases for backward compatibility functions."""

    def test_append_chat_message_compatibility(self, mock_supabase, sample_message_data):
        """Test that append_chat_message still works (backward compatibility)."""
        # Mock the insert operation for messages
        mock_insert_result = Mock()
        mock_insert_result.data = [sample_message_data]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        # Mock the chat metadata update (count query and update)
        mock_count_result = Mock()
        mock_count_result.count = 1
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()

        # Execute function using the backward-compatible interface
        message_input = ChatMessageInput(
            chat_id="chat-123", role="user", content="New message"
        )
        result = append_chat_message(mock_supabase, message_input)

        # Verify result is a Message (new system returns Message, not Chat)
        assert isinstance(result, Message)
        assert result.id == sample_message_data["id"]
        assert result.content == "New message"

    def test_append_chat_message_with_optional_fields(self, mock_supabase, sample_message_data):
        """Test appending a message with optional fields."""
        mock_insert_result = Mock()
        mock_insert_result.data = [sample_message_data]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_insert_result

        # Mock the chat metadata update
        mock_count_result = Mock()
        mock_count_result.count = 1
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_count_result
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock()

        message_input = ChatMessageInput(
            chat_id="chat-123",
            role="user", 
            content="New message",
            message_type="chat",
            task_group_id="group-123"
        )
        result = append_chat_message(mock_supabase, message_input)

        # Verify the function completed successfully with optional fields
        assert isinstance(result, Message)
        assert result.id == sample_message_data["id"]

    def test_append_chat_message_database_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("DB Error")

        message_input = ChatMessageInput(
            chat_id="chat-123", role="user", content="message"
        )
        
        with pytest.raises(Exception, match="DB Error"):
            append_chat_message(mock_supabase, message_input)
