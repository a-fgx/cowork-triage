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
### [Agent] - 2026-01-23 16:05:16
**Classification:**
- Type: `runtime`
- Confidence: 90%
- Reasoning: The error message `ValueError: ToolStrategy specifies tool 'response_format_5549' which wasn't declared in the original response format when creating the agent.` indicates a runtime error during agent execution. The `AutoStrategy` is dynamically creating a `ToolStrategy` with a new tool name that wasn't pre-defined, leading to the `ValueError`. This suggests a problem with how the agent and its tools are being initialized and managed during the execution flow, specifically within the `langchain.agents` module.


# Diagnostic Report

## Detected Libraries
**Primary:** langchain
**Also involved:** langgraph, langsmith
**Components:** ChatOpenAI, langchain_openai, langchain, trace, create_agent

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
**Root Cause:** The `AutoStrategy` in `langchain.agents.structured_output` is dynamically creating a `ToolStrategy` with a new tool name that is not pre-declared in the agent's tool list. The `create_agent` function expects all tools to be defined upfront, but `AutoStrategy` introduces a new tool at runtime, leading to the `ValueError`. This is likely a bug in how `AutoStrategy` interacts with the agent creation process.
**Confidence:** High

### Confidence Sources
| Source | Score |
|--------|-------|
| LLM Classification | 90% |
| GitHub Issues | 68% |
| RAG Knowledge Base | 82% |
| Library Detection | 100% |
| **Overall** | **82%** |

*Main contributors: LLM classification: 90%, GitHub issue matches: 68%*

**Supporting Evidence:**
- Error message: `ValueError: ToolStrategy specifies tool 'response_format_5549' which wasn't declared in the original response format when creating the agent.`
- The code uses `AutoStrategy` which is designed to dynamically select a response format.
- The error occurs during agent invocation, suggesting a runtime issue.
- Related GitHub issue #34796 confirms this is a bug with AutoStrategy and ToolStrategy.

## Resolution Plan

### Step 1: Upgrade `langchain` to the latest version: `pip install -U langchain`
*Why:* The error is related to the `AutoStrategy` and `ToolStrategy` interaction within `langchain.agents.structured_output`. Upgrading to the latest version ensures that you have the most recent bug fixes and improvements related to these components. While there isn't a specific fix mentioned, upgrading is a general first step to rule out outdated code.
*Expected result:* The `langchain` package is updated to the latest available version.

### Step 2: Explicitly define the tools used by the agent instead of relying on `AutoStrategy` to dynamically create them. Replace `agent = create_agent(model, response_format=AutoStrategy(schema))` with a tool-based agent creation.  First, define a tool that uses the schema.  Then, pass that tool to `create_agent`.  For example:

```python
from langchain.tools import StructuredTool

def process_result(result: str) -> dict:
    return {"result": result}

result_tool = StructuredTool.from_function(process_result)

agent = create_agent(model, tools=[result_tool])
```
*Why:* The error indicates that the `ToolStrategy` is creating a tool at runtime that isn't declared upfront. By explicitly defining the tool and passing it to `create_agent`, we avoid the dynamic tool creation and ensure that all tools are known at agent creation time. This bypasses the problematic `AutoStrategy` logic.
*Expected result:* The agent is created successfully without the `ValueError` related to undeclared tools.

### Step 3: If the previous step resolves the immediate error but introduces new issues with how the agent handles the schema, consider using a custom agent that directly incorporates the schema into the prompt. This involves creating a custom prompt template that includes the schema and then using `LLMChain` to chain the model and prompt. This approach gives you more control over how the schema is used and avoids the complexities of `AutoStrategy` and `ToolStrategy`.
*Why:* While defining the tool explicitly should resolve the error, it might not fully replicate the intended behavior of `AutoStrategy`. If the agent's behavior is not as expected, a custom agent provides more control over the interaction between the model and the schema.
*Expected result:* The agent processes the input and returns a result conforming to the defined schema, with more control over the agent's behavior.

---
*If the issue persists after following these steps, please provide additional details.*
