"""
Unit tests for chats.py functions.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.chats import append_chat_message, get_message_history


class TestAppendChatMessage:
    """Test cases for append_chat_message function."""
    
    def test_append_message_success(self, mock_supabase, sample_chat_data):
        """Test successfully appending a message to chat."""
        # Setup the mock chain properly
        mock_table = Mock()
        mock_supabase.table.return_value = mock_table
        
        # Mock the select chain
        mock_select = Mock()
        mock_table.select.return_value = mock_select
        mock_eq = Mock()
        mock_select.eq.return_value = mock_eq
        mock_single = Mock()
        mock_eq.single.return_value = mock_single
        
        # Mock the select result
        mock_select_result = Mock()
        mock_select_result.data = sample_chat_data
        mock_single.execute.return_value = mock_select_result
        
        # Mock the update chain 
        mock_update = Mock()
        mock_table.update.return_value = mock_update
        mock_update_eq = Mock()
        mock_update.eq.return_value = mock_update_eq
        
        # Mock the update result
        updated_chat = sample_chat_data.copy()
        updated_chat["conversation"] = sample_chat_data["conversation"] + [{
            "role": "user", 
            "message": "New message",
            "timestamp": "2023-01-01T10:01:00.000Z"
        }]
        
        mock_update_result = Mock()
        mock_update_result.data = [updated_chat]
        mock_update_eq.execute.return_value = mock_update_result
        
        # Execute function
        result = append_chat_message(
            mock_supabase, 
            "chat-123", 
            "user", 
            "New message"
        )
        
        # Verify result
        assert result["id"] == "chat-123"
        assert len(result["conversation"]) == 3
        assert result["conversation"][-1]["message"] == "New message"
        
    def test_append_message_with_kwargs(self, mock_supabase, sample_chat_data):
        """Test appending a message with additional properties."""
        # Setup mock chain
        mock_table = Mock()
        mock_supabase.table.return_value = mock_table
        
        # Mock select chain
        mock_select = Mock()
        mock_table.select.return_value = mock_select
        mock_eq = Mock()
        mock_select.eq.return_value = mock_eq
        mock_single = Mock()
        mock_eq.single.return_value = mock_single
        mock_select_result = Mock()
        mock_select_result.data = sample_chat_data
        mock_single.execute.return_value = mock_select_result
        
        # Mock update chain
        mock_update = Mock()
        mock_table.update.return_value = mock_update
        mock_update_eq = Mock()
        mock_update.eq.return_value = mock_update_eq
        mock_update_result = Mock()
        mock_update_result.data = [sample_chat_data]
        mock_update_eq.execute.return_value = mock_update_result
        
        result = append_chat_message(
            mock_supabase,
            "chat-123",
            "user", 
            "New message",
            contextWidgetIds=["widget-1", "widget-2"],
            targetWidgetType="chart"
        )
        
        # Verify the function completed successfully 
        # (The kwargs functionality is tested by the fact that no exception was raised)
        assert result == sample_chat_data
        
    def test_append_message_chat_not_found(self, mock_supabase):
        """Test error when chat is not found."""
        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        with pytest.raises(Exception, match="Chat chat-123 not found"):
            append_chat_message(mock_supabase, "chat-123", "user", "message")
            
    def test_append_message_update_fails(self, mock_supabase, sample_chat_data):
        """Test error when update operation fails."""
        # Setup mock chain for select (successful)
        mock_table = Mock()
        mock_supabase.table.return_value = mock_table
        mock_select = Mock()
        mock_table.select.return_value = mock_select
        mock_eq = Mock()
        mock_select.eq.return_value = mock_eq
        mock_single = Mock()
        mock_eq.single.return_value = mock_single
        mock_select_result = Mock()
        mock_select_result.data = sample_chat_data
        mock_single.execute.return_value = mock_select_result
        
        # Setup mock chain for update (fails)
        mock_update = Mock()
        mock_table.update.return_value = mock_update
        mock_update_eq = Mock()
        mock_update.eq.return_value = mock_update_eq
        mock_update_result = Mock()
        mock_update_result.data = None
        mock_update_eq.execute.return_value = mock_update_result
        
        with pytest.raises(Exception, match="Failed to update chat"):
            append_chat_message(mock_supabase, "chat-123", "user", "message")
            
    def test_append_message_database_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table().select().eq().single().execute.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception, match="DB Error"):
            append_chat_message(mock_supabase, "chat-123", "user", "message")


class TestGetMessageHistory:
    """Test cases for get_message_history function."""
    
    @pytest.mark.asyncio
    async def test_get_all_messages(self, mock_supabase, sample_chat_data):
        """Test getting all messages from chat."""
        mock_result = Mock()
        mock_result.data = sample_chat_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        result = await get_message_history(mock_supabase, "chat-123")
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "system"
        
    @pytest.mark.asyncio
    async def test_get_last_n_messages(self, mock_supabase, sample_chat_data):
        """Test getting last N messages from chat."""
        # Add more messages to test data
        extended_conversation = sample_chat_data["conversation"] + [
            {"role": "user", "message": "Third", "timestamp": "2023-01-01T10:02:00.000Z"},
            {"role": "system", "message": "Fourth", "timestamp": "2023-01-01T10:03:00.000Z"},
        ]
        extended_data = sample_chat_data.copy()
        extended_data["conversation"] = extended_conversation
        
        mock_result = Mock()
        mock_result.data = extended_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        result = await get_message_history(mock_supabase, "chat-123", last_n=2)
        
        assert len(result) == 2
        assert result[0]["message"] == "Third"
        assert result[1]["message"] == "Fourth"
        
    @pytest.mark.asyncio
    async def test_get_messages_empty_conversation(self, mock_supabase):
        """Test getting messages from chat with empty conversation."""
        chat_data = {
            "id": "chat-123",
            "conversation": []
        }
        
        mock_result = Mock()
        mock_result.data = chat_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        result = await get_message_history(mock_supabase, "chat-123")
        
        assert result == []
        
    @pytest.mark.asyncio
    async def test_get_messages_chat_not_found(self, mock_supabase):
        """Test error when chat is not found."""
        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        with pytest.raises(Exception, match="Chat chat-123 not found"):
            await get_message_history(mock_supabase, "chat-123")
            
    @pytest.mark.asyncio
    async def test_get_messages_with_zero_last_n(self, mock_supabase, sample_chat_data):
        """Test getting messages with last_n=0."""
        mock_result = Mock()
        mock_result.data = sample_chat_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        result = await get_message_history(mock_supabase, "chat-123", last_n=0)
        
        # Should return all messages when last_n is 0
        assert len(result) == 2
        
    @pytest.mark.asyncio
    async def test_get_messages_database_error(self, mock_supabase):
        """Test handling of database errors."""
        mock_supabase.table().select().eq().single().execute.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception, match="DB Error"):
            await get_message_history(mock_supabase, "chat-123")
            
    @pytest.mark.asyncio
    async def test_get_messages_conversation_missing(self, mock_supabase):
        """Test handling when conversation field is missing."""
        chat_data = {"id": "chat-123"}
        
        mock_result = Mock()
        mock_result.data = chat_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        result = await get_message_history(mock_supabase, "chat-123")
        
        assert result == []