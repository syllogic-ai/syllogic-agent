"""Message management utilities for Supabase messages table.
Provides functions to create and manage individual chat messages.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from supabase import Client

from agent.models import Message, CreateMessageInput

logger = logging.getLogger(__name__)


def create_message(supabase: Client, message_input: CreateMessageInput) -> Message:
    """Create a new message in the database.

    Args:
        supabase: Supabase client instance
        message_input: CreateMessageInput containing message data

    Returns:
        Message model with the created message data

    Raises:
        Exception: If message creation fails
    """
    try:
        # Generate unique message ID
        message_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        # Create message data
        message_data = {
            "id": message_id,
            "chat_id": message_input.chat_id,
            "role": message_input.role,
            "content": message_input.content,
            "message_type": message_input.message_type,
            "task_group_id": message_input.task_group_id,
            "created_at": current_time,
            "updated_at": current_time,
        }

        # Insert message into database
        result = supabase.table("messages").insert(message_data).execute()

        if result.data:
            logger.info(f"Created message {message_id} in chat {message_input.chat_id}")
            
            # Update chat metadata
            _update_chat_metadata(supabase, message_input.chat_id)
            
            return Message(**result.data[0])
        else:
            raise Exception(f"Failed to create message: No data returned")

    except Exception as e:
        logger.error(f"Error creating message in chat {message_input.chat_id}: {str(e)}")
        raise


def create_task_list_message(
    supabase: Client, 
    chat_id: str, 
    task_group_id: str, 
    task_list_content: str
) -> Message:
    """Create a task-list type message in the database.

    Args:
        supabase: Supabase client instance
        chat_id: Chat ID to create message in
        task_group_id: Task group ID to link with tasks
        task_list_content: Formatted task list content

    Returns:
        Message model with the created message data

    Raises:
        Exception: If message creation fails
    """
    try:
        message_input = CreateMessageInput(
            chat_id=chat_id,
            role="ai",  # Correct role as specified
            content=task_list_content,  # Add the content parameter
            message_type="task-list",
            task_group_id=task_group_id
        )

        return create_message(supabase, message_input)

    except Exception as e:
        logger.error(f"Error creating task list message: {str(e)}")
        raise


def get_messages_by_chat(
    supabase: Client, 
    chat_id: str, 
    limit: Optional[int] = None
) -> List[Message]:
    """Get messages from a chat, ordered by creation time.

    Args:
        supabase: Supabase client instance
        chat_id: Chat ID to get messages from
        limit: Optional limit on number of messages to return

    Returns:
        List of Message models ordered by creation time

    Raises:
        Exception: If retrieval fails
    """
    try:
        query = (
            supabase.table("messages")
            .select("*")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
        )

        if limit:
            query = query.limit(limit)

        result = query.execute()

        messages = [Message(**msg_data) for msg_data in result.data]
        logger.info(f"Retrieved {len(messages)} messages from chat {chat_id}")
        return messages

    except Exception as e:
        logger.error(f"Error retrieving messages from chat {chat_id}: {str(e)}")
        raise


def get_messages_by_task_group(
    supabase: Client, 
    task_group_id: str
) -> List[Message]:
    """Get messages linked to a specific task group.

    Args:
        supabase: Supabase client instance
        task_group_id: Task group ID to filter by

    Returns:
        List of Message models linked to the task group

    Raises:
        Exception: If retrieval fails
    """
    try:
        result = (
            supabase.table("messages")
            .select("*")
            .eq("task_group_id", task_group_id)
            .order("created_at", desc=False)
            .execute()
        )

        messages = [Message(**msg_data) for msg_data in result.data]
        logger.info(f"Retrieved {len(messages)} messages for task group {task_group_id}")
        return messages

    except Exception as e:
        logger.error(f"Error retrieving messages for task group {task_group_id}: {str(e)}")
        raise


def _update_chat_metadata(supabase: Client, chat_id: str) -> None:
    """Update chat metadata (message count, last message time).

    Args:
        supabase: Supabase client instance
        chat_id: Chat ID to update

    Raises:
        Exception: If update fails
    """
    try:
        current_time = datetime.now().isoformat()

        # Get current message count
        count_result = (
            supabase.table("messages")
            .select("id", count="exact")
            .eq("chat_id", chat_id)
            .execute()
        )

        message_count = count_result.count or 0

        # Update chat metadata
        supabase.table("chats").update({
            "message_count": message_count,
            "last_message_at": current_time,
            "updated_at": current_time
        }).eq("id", chat_id).execute()

        logger.debug(f"Updated chat {chat_id} metadata: {message_count} messages")

    except Exception as e:
        logger.warning(f"Failed to update chat {chat_id} metadata: {str(e)}")
        # Don't raise - this is non-critical


def append_chat_message(supabase: Client, message_input: CreateMessageInput) -> Message:
    """Create a new chat message (wrapper for create_message for backward compatibility).

    Args:
        supabase: Supabase client instance
        message_input: CreateMessageInput containing message data

    Returns:
        Message model with the created message data

    Raises:
        Exception: If message creation fails
    """
    return create_message(supabase, message_input)