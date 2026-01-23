"""
LLM Integration Module

This module provides a configured Gemini client for use across the agent.
It wraps LangChain's Google Generative AI integration.

Key Concept: Structured Output
The LLM can return structured data (dicts) instead of plain text by using
the `with_structured_output()` method. This is crucial for parsing responses
into our TypedDict schemas.
"""

from typing import Type, TypeVar
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config

# Type variable for structured output
T = TypeVar("T")


def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
) -> ChatGoogleGenerativeAI:
    """
    Get a configured Gemini chat model.

    Args:
        model: Model name (defaults to config.gemini_model)
               Options: "gemini-1.5-flash" (fast), "gemini-1.5-pro" (quality)
        temperature: Randomness (0.0 = deterministic, 1.0 = creative)

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Example:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content="Hello!")])
        print(response.content)
    """
    return ChatGoogleGenerativeAI(
        model=model or config.gemini_model,
        google_api_key=config.google_api_key,
        temperature=temperature,
        # Convert Gemini's blocked responses to exceptions
        # (instead of returning empty content)
        convert_system_message_to_human=True,
    )


def invoke_with_system(
    llm: ChatGoogleGenerativeAI,
    system_prompt: str,
    user_message: str,
) -> str:
    """
    Invoke the LLM with a system prompt and user message.

    This is a convenience wrapper for simple request-response patterns.

    Args:
        llm: The LLM instance
        system_prompt: Instructions for how the LLM should behave
        user_message: The actual query/input

    Returns:
        The LLM's response as a string

    Example:
        llm = get_llm()
        response = invoke_with_system(
            llm,
            system_prompt="You are a helpful assistant.",
            user_message="What is 2+2?"
        )
        print(response)  # "4"
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response.content


def invoke_structured(
    llm: ChatGoogleGenerativeAI,
    system_prompt: str,
    user_message: str,
    output_schema: Type[T],
) -> T:
    """
    Invoke the LLM and parse the response into a structured schema.

    This uses LangChain's structured output feature to ensure the response
    matches the expected TypedDict or Pydantic model.

    Args:
        llm: The LLM instance
        system_prompt: Instructions for how the LLM should behave
        user_message: The actual query/input
        output_schema: The TypedDict or Pydantic model to parse into

    Returns:
        The parsed response matching the schema

    Example:
        from src.state import Classification

        llm = get_llm()
        result = invoke_structured(
            llm,
            system_prompt="Classify this bug report.",
            user_message="My API call returns 500...",
            output_schema=Classification
        )
        print(result["failure_type"])  # "api"
        print(result["confidence"])    # 0.85
    """
    # Create a structured LLM that outputs the schema
    structured_llm = llm.with_structured_output(output_schema)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    return structured_llm.invoke(messages)


# =============================================================================
# Pre-configured LLM instances
# =============================================================================

# Default LLM for general use (fast model)
llm = get_llm()

# High-quality LLM for complex reasoning (slower but better)
llm_pro = get_llm(model="gemini-2.5-pro-preview-05-06")
