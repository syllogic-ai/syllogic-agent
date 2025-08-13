"""Chat management utilities for Supabase chats table.
Provides functions to append chat messages to conversations.
"""

import logging
from datetime import datetime
from typing import List, Optional

from supabase import Client

from agent.models import Chat, ChatMessageInput, ConversationMessage

logger = logging.getLogger(__name__)


def append_chat_message(supabase: Client, message_input: ChatMessageInput) -> Chat:
    """Append a chat message to the conversation array in Supabase.

    Args:
        supabase: Supabase client instance
        message_input: ChatMessageInput containing message data

    Returns:
        Chat model with the updated chat data

    Raises:
        Exception: If message append fails
    """
    try:
        # First, get the current conversation
        current_chat = (
            supabase.table("chats")
            .select("conversation")
            .eq("id", message_input.chat_id)
            .single()
            .execute()
        )

        if not current_chat.data:
            raise Exception(f"Chat {message_input.chat_id} not found")

        # Get current conversation array
        current_conversation = current_chat.data.get("conversation", [])

        # Create new message object
        new_message = ConversationMessage(
            role=message_input.role,
            message=message_input.message,
            timestamp=datetime.now().isoformat(),
            context_widget_ids=message_input.context_widget_ids,
            target_widget_type=message_input.target_widget_type,
            target_chart_sub_type=message_input.target_chart_sub_type,
        ).model_dump(exclude_none=True)

        # Append the new message to the conversation
        updated_conversation = current_conversation + [new_message]

        # Update the chat with the new conversation
        result = (
            supabase.table("chats")
            .update(
                {
                    "conversation": updated_conversation,
                    "updated_at": datetime.now().isoformat(),
                }
            )
            .eq("id", message_input.chat_id)
            .execute()
        )

        if result.data:
            logger.info(
                f"Appended {message_input.role} message to chat {message_input.chat_id}"
            )
            return Chat(**result.data[0])
        else:
            raise Exception(
                f"Failed to update chat {message_input.chat_id}: No data returned"
            )

    except Exception as e:
        logger.error(
            f"Error appending message to chat {message_input.chat_id}: {str(e)}"
        )
        raise


async def get_message_history(
    supabase: Client, chat_id: str, last_n: Optional[int] = None
) -> List[ConversationMessage]:
    """Get message history from the conversation column of a chat.

    Args:
        supabase: Supabase client instance
        chat_id: Chat identifier
        last_n: Optional number of recent messages to return. If None, returns all messages.

    Returns:
        List of ConversationMessage models from the conversation array

    Raises:
        Exception: If chat not found or retrieval fails
    """
    try:
        # Get the conversation from the chat
        chat_result = (
            supabase.table("chats")
            .select("conversation")
            .eq("id", chat_id)
            .single()
            .execute()
        )

        if not chat_result.data:
            raise Exception(f"Chat {chat_id} not found")

        # Get the conversation array
        conversation = chat_result.data.get("conversation", [])

        # If last_n is specified, return only the last N messages
        if last_n is not None and last_n > 0:
            conversation = conversation[-last_n:]

        # Convert to ConversationMessage models
        messages = [ConversationMessage(**msg) for msg in conversation]
        logger.info(f"Retrieved {len(messages)} messages from chat {chat_id}")
        return messages

    except Exception as e:
        logger.error(f"Error getting message history from chat {chat_id}: {str(e)}")
        raise
