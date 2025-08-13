"""
Chat management utilities for Supabase chats table.
Provides functions to append chat messages to conversations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from supabase import Client
import logging

logger = logging.getLogger(__name__)


def append_chat_message(
    supabase: Client,
    chat_id: str,
    role: str,
    message: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Append a chat message to the conversation array in Supabase.
    
    Args:
        supabase: Supabase client instance
        chat_id: Chat identifier
        role: Message role ('user', 'system', 'assistant')
        message: Message content
        **kwargs: Additional message properties (contextWidgetIds, targetWidgetType, etc.)
        
    Returns:
        Dict containing the updated chat data
        
    Raises:
        Exception: If message append fails
    """
    try:
        # First, get the current conversation
        current_chat = supabase.table("chats").select("conversation").eq("id", chat_id).single().execute()
        
        if not current_chat.data:
            raise Exception(f"Chat {chat_id} not found")
        
        # Get current conversation array
        current_conversation = current_chat.data.get("conversation", [])
        
        # Create new message object
        new_message = {
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add any additional properties from kwargs
        for key, value in kwargs.items():
            if value is not None:
                new_message[key] = value
        
        # Append the new message to the conversation
        updated_conversation = current_conversation + [new_message]
        
        # Update the chat with the new conversation
        result = supabase.table("chats").update({
            "conversation": updated_conversation,
            "updated_at": datetime.now().isoformat()
        }).eq("id", chat_id).execute()
        
        if result.data:
            logger.info(f"Appended {role} message to chat {chat_id}")
            return result.data[0]
        else:
            raise Exception(f"Failed to update chat {chat_id}: No data returned")
            
    except Exception as e:
        logger.error(f"Error appending message to chat {chat_id}: {str(e)}")
        raise


async def get_message_history(
    supabase: Client,
    chat_id: str,
    last_n: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get message history from the conversation column of a chat.
    
    Args:
        supabase: Supabase client instance
        chat_id: Chat identifier
        last_n: Optional number of recent messages to return. If None, returns all messages.
        
    Returns:
        List of message dictionaries from the conversation array
        
    Raises:
        Exception: If chat not found or retrieval fails
    """
    try:
        # Get the conversation from the chat
        chat_result = supabase.table("chats").select("conversation").eq("id", chat_id).single().execute()
        
        if not chat_result.data:
            raise Exception(f"Chat {chat_id} not found")
        
        # Get the conversation array
        conversation = chat_result.data.get("conversation", [])
        
        # If last_n is specified, return only the last N messages
        if last_n is not None and last_n > 0:
            conversation = conversation[-last_n:]
        
        logger.info(f"Retrieved {len(conversation)} messages from chat {chat_id}")
        return conversation
        
    except Exception as e:
        logger.error(f"Error getting message history from chat {chat_id}: {str(e)}")
        raise