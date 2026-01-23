# Ticket 001

## Status: Open

## Title
Unable to use agent response format when AutoStrategy selects ToolStrategy

## Description
When using AutoStrategy, the effective_response_format is a newly created ToolStrategy object with a new tool name, but structured_output_tools are declared upfront.

## Reproduction Steps / Example Code (Python)
import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy


model = ChatOpenAI(
    model="google/gemini-2.5-pro",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


schema = {
    "type": "object",
    "required": ["result"],
    "properties": {
        "result": {
            "type": "string",
        }
    },
}

agent = create_agent(model, response_format=AutoStrategy(schema))

result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
print(result)

## Error Message and Stack Trace (if applicable)
ValueError: ToolStrategy specifies tool 'response_format_5549' which wasn't declared in the original response format when creating the agent.
During task with name 'model' and id '3453861c-6038-975b-db82-704ccb42d096'

## Exchange Log
### [Agent] - 2026-01-23 14:53:04
**Classification:**
- Type: `runtime`
- Confidence: 90%
- Reasoning: The error message `ValueError: ToolStrategy specifies tool 'response_format_5549' which wasn't declared in the original response format when creating the agent.` indicates a runtime error during agent execution. The `AutoStrategy` is dynamically creating a `ToolStrategy` with a new tool name that wasn't pre-defined, leading to the `ValueError`. This suggests a problem with how the agent and its tools are being initialized and managed during the execution flow, specifically within the `langchain.agents` module.


# Diagnostic Report

## Detected Libraries
**Primary:** langchain
**Also involved:** langgraph, langsmith
**Components:** ChatOpenAI, langchain_openai, create_agent, langchain, trace

## Similar GitHub Issues Found
- **[#34796: Unable to use agent response format when AutoStrategy selects ToolStrategy](https://github.com/langchain-ai/langchain/issues/34796)** [langchain] (🔴 open)
  > ### Checked other resources  - [x] This is a bug, not a usage question. - [x] I added a clear and descriptive title that summarizes this issue. - [x] ...
- **[#34797: Unable to use agent response format with ProviderStrategy and ChatOpenAI](https://github.com/langchain-ai/langchain/issues/34797)** [langchain] (🔴 open)
  > ### Checked other resources  - [x] This is a bug, not a usage question. - [x] I added a clear and descriptive title that summarizes this issue. - [x] ...
- **[#6675: calling `model_dump()` on pydantic state variables drops `tool_calls` in AIMessage](https://github.com/langchain-ai/langgraph/issues/6675)** [langgraph] (🔴 open)
  > ### Checked other resources  - [x] This is a bug, not a usage question. For questions, please use the LangChain Forum (https://forum.langchain.com/). ...
- **[#2082: Does this example with FastAPI + LangSmith tracing actually work? (FastAPI [standard] 0.119.1+, LangChain 1.0.1+, LangSmith 0.4.37+)](https://github.com/langchain-ai/langsmith-sdk/issues/2082)** [langsmith-sdk] (✅ closed)
  > I am trying to set up tracing with LangSmith in a FastAPI app and I’d like to check if the following minimal example *should work*, or whether I am mi...
- **[#34239: Dynamic response_format with create_agent](https://github.com/langchain-ai/langchain/issues/34239)** [langchain] (🔴 open)
  > ### Checked other resources  - [x] This is a feature request, not a bug report or usage question. - [x] I added a clear and descriptive title that sum...

## Diagnosis
**Root Cause:** The `AutoStrategy` in `langchain.agents.structured_output` is dynamically creating a `ToolStrategy` with a new tool name that is not pre-declared in the original response format. The `create_agent` function expects all tools to be defined upfront, but `AutoStrategy` violates this expectation by introducing a new tool at runtime. This leads to a mismatch between the expected and actual tool configurations, resulting in the `ValueError`.
**Confidence:** High

### Confidence Sources
| Source | Score |
|--------|-------|
| LLM Classification | 90% |
| GitHub Issues | 0% |
| RAG Knowledge Base | 82% |
| Library Detection | 100% |
| **Overall** | **58%** |

*Main contributors: LLM classification: 90%, RAG knowledge base: 82%*

**Supporting Evidence:**
- Error message: `ValueError: ToolStrategy specifies tool 'response_format_5549' which wasn't declared in the original response format when creating the agent.`
- The code uses `AutoStrategy` which is designed to dynamically select a response format.
- The error occurs during agent invocation, suggesting a runtime issue with tool selection.
- Related GitHub issue #34796: Unable to use agent response format when AutoStrategy selects ToolStrategy (open)

## Resolution Plan

### Step 1: Review the error message and stack trace carefully
*Why:* Understanding the exact error is the first step
*Expected result:* Identify the specific line or function causing the issue

### Step 2: Search for the error message online
*Why:* Others may have encountered and solved this issue
*Expected result:* Find relevant Stack Overflow posts or documentation

---
*If the issue persists after following these steps, please provide additional details.*
