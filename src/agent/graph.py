"""LangGraph chatbot with OpenAI GPT-4o-mini integration.

A minimal chatbot that processes user messages and responds using OpenAI's API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from openai import AsyncOpenAI
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime


class Context(TypedDict):
    """Context parameters for the chatbot.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    openai_api_key: str
    model: str


@dataclass
class State:
    """Input state for the chatbot.

    Defines the structure for chat messages and responses.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    user_message: str = ""
    assistant_response: str = ""
    conversation_history: list = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process user message and generate response using OpenAI GPT-4o-mini.

    Uses OpenAI's API to generate contextual responses based on conversation history.
    """
    # Get configuration from runtime context with null checks
    context = runtime.context or {}
    api_key = context.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
    model = context.get('model', 'gpt-4o-mini')
    
    if not api_key:
        return {
            "assistant_response": "Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide it in context.",
            "conversation_history": state.conversation_history
        }
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    
    # Build messages for OpenAI API
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide clear, concise, and helpful responses."}
    ]
    
    # Add conversation history
    for msg in state.conversation_history:
        messages.append(msg)
    
    # Add current user message
    if state.user_message:
        messages.append({"role": "user", "content": state.user_message})
    
    try:
        # Call OpenAI API
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        # Update conversation history
        updated_history = state.conversation_history.copy()
        if state.user_message:
            updated_history.append({"role": "user", "content": state.user_message})
        updated_history.append({"role": "assistant", "content": assistant_response})
        
        return {
            "assistant_response": assistant_response,
            "conversation_history": updated_history
        }
        
    except Exception as e:
        return {
            "assistant_response": f"Error calling OpenAI API: {str(e)}",
            "conversation_history": state.conversation_history
        }
    finally:
        await client.close()


# Define the chatbot graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="OpenAI Chatbot")
)
