# Ticket 002

## Status: Open

## Title
Tool calls sometimes fail due to missing positional argument

## Description
I am trying to use LangChain agents to assist in putting together a job plan / configuration for an HPC application. The simple tool call above consistently fails due to its positional argument being missing. By wrapping the tool call I confirm that the argument (config.0) is actually being passed to the tool correctly so the issue is happening at a lower level within the tool execution chain:

ToolCallRequest(tool_call={'name': 'addConfig', 'args': {'config': 'config.0'}, 'id': 'WnemfvoElnQZJnjhgfT7wlBvzWm1jNXt', 'type': 'tool_call'}, tool=StructuredTool(name='addConfig', description='Add a configuration file name to the list of configurations\nArgs:\n   config: The file name, formatted as a path (relative or absolute)', args_schema=<class 'langchain_core.utils.pydantic.addConfig'>, func=<function addConfig at 0x7ff9bb3d67a0>), state=...
The issue appears to go away if the argument is renamed. For example, with

def addConfig(configf: str, runtime: ToolRuntime):
    """                                                                                                                                                                                                                                                              
    Add a configuration file name to the list of configurations                                                                                                                                                                                                      
    Args:                                                                                                                                                                                                                                                            
       configf: The file name, formatted as a path (relative or absolute)                                                                                                                                                                                             
    """
    print("(Dummy) Appending configuration ",configf)
the tool is executed correctly:

ToolCallRequest(tool_call={'name': 'addConfig', 'args': {'configf': 'config.0'}, 'id': 'L3kzDmy7D8bN0NtYdjTEcOXOlqqe4NQQ', 'type': 'tool_call'}, ...
...
...
(Dummy) Appending configuration  config.0
This suggests some sort of keyword / name collision somewhere within the bowels of the software.

## Reproduction Steps / Example Code (Python)
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_tool_call, AgentState

llm = ChatOpenAI(
    model="Qwen3-8B-GGUF",
    openai_api_key="sk-local",  # Or a placeholder if required by the local server                                                                                                                                                                                   
    openai_api_base="http://localhost:8002/v1"
)

@tool
def addConfig(config: str, runtime: ToolRuntime):
    """                                                                                                                                                                                                                                                              
    Add a configuration file name to the list of configurations                                                                                                                                                                                                      
    Args:                                                                                                                                                                                                                                                            
       config: The file name, formatted as a path (relative or absolute)                                                                                                                                                                                             
    """
    print("(Dummy) Appending configuration ",config)

@wrap_tool_call
def printToolCalls(request, handler):
    print(request)
    return handler(request)

agent = create_agent(
    model=llm,
    tools=[addConfig],
    system_prompt="You are a helpful assistant.",
    middleware = [printToolCalls]
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Add the configuration config.0"}]}
)

## Error Message and Stack Trace (if applicable)
Traceback (most recent call last):
  File "/home/ckelly/test/langchain/qcd_workflow/config_state/weird_err_reproducer/repro.py", line 36, in <module>
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Add the configuration config.0"}]}
    )
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/pregel/main.py", line 3050, in invoke
    for chunk in self.stream(
                 ~~~~~~~~~~~^
        input,
        ^^^^^^
    ...<10 lines>...
        **kwargs,
        ^^^^^^^^^
    ):
    ^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/pregel/main.py", line 2633, in stream
    for _ in runner.tick(
             ~~~~~~~~~~~^
        [t for t in loop.tasks.values() if not t.writes],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        schedule_task=loop.accept_push,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/pregel/_runner.py", line 167, in tick
    run_with_retry(
    ~~~~~~~~~~~~~~^
        t,
        ^^
    ...<10 lines>...
        },
        ^^
    )
    ^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/pregel/_retry.py", line 42, in run_with_retry
    return task.proc.invoke(task.input, config)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/_internal/_runnable.py", line 656, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/_internal/_runnable.py", line 400, in invoke
    ret = self.func(*args, **kwargs)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 727, in _func
    outputs = list(
        executor.map(self._run_one, tool_calls, input_types, tool_runtimes)
    )
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/concurrent/futures/_base.py", line 619, in result_iterator
    yield _result_or_cancel(fs.pop())
          ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/concurrent/futures/_base.py", line 317, in _result_or_cancel
    return fut.result(timeout)
           ~~~~~~~~~~^^^^^^^^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ~~~~~~~~~~~~~~~~~^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langchain_core/runnables/config.py", line 546, in _wrapped_fn
    return contexts.pop().run(fn, *args)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 957, in _run_one
    content = _handle_tool_error(e, flag=self._handle_tool_errors)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 404, in _handle_tool_error
    content = flag(e)  # type: ignore [assignment, call-arg]
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 361, in _default_handle_tool_errors
    raise e
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 951, in _run_one
    return self._wrap_tool_call(tool_request, execute)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langchain/agents/middleware/types.py", line 1604, in wrapped
    return func(request, handler)
  File "/home/ckelly/test/langchain/qcd_workflow/config_state/weird_err_reproducer/repro.py", line 27, in printToolCalls
    return handler(request)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 947, in execute
    return self._execute_tool_sync(req, input_type, config)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 891, in _execute_tool_sync
    content = _handle_tool_error(e, flag=self._handle_tool_errors)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 404, in _handle_tool_error
    content = flag(e)  # type: ignore [assignment, call-arg]
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 361, in _default_handle_tool_errors
    raise e
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langgraph/prebuilt/tool_node.py", line 844, in _execute_tool_sync
    response = tool.invoke(call_args, config)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langchain_core/tools/base.py", line 605, in invoke
    return self.run(tool_input, **kwargs)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langchain_core/tools/base.py", line 932, in run
    raise error_to_raise
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langchain_core/tools/base.py", line 898, in run
    response = context.run(self._run, *tool_args, **tool_kwargs)
  File "/home/ckelly/miniconda3/envs/ai-py313/lib/python3.13/site-packages/langchain_core/tools/structured.py", line 93, in _run
    return self.func(*args, **kwargs)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^
TypeError: addConfig() missing 1 required positional argument: 'config'
During task with name 'tools' and id '947599f3-98b7-5563-3987-38d0d47f7223'

## Exchange Log
### [Agent] - 2026-01-22 12:06:10
**Classification:**
- Type: `runtime`
- Confidence: 90%
- Reasoning: The stack trace clearly shows a `TypeError: addConfig() missing 1 required positional argument: 'config'`. This indicates a failure during the execution of the tool, specifically when calling the `addConfig` function. The issue seems to stem from how the arguments are being passed to the tool's function within the LangGraph execution framework, particularly within `langgraph/prebuilt/tool_node.py` and `langchain_core/tools/structured.py`. The fact that renaming the argument fixes the issue suggests a potential name collision or incorrect argument mapping within the LangChain/LangGraph internals.


# Diagnostic Report

## Detected Libraries
**Primary:** langchain
**Also involved:** langgraph, langsmith
**Components:** AgentState, langchain, trace, @tool, langchain_openai

## Similar GitHub Issues Found
- **[#34029: Tool calls sometimes fail due to missing positional argument](https://github.com/langchain-ai/langchain/issues/34029)** [langchain] (🔴 open)
  > ### Checked other resources  - [x] This is a bug, not a usage question. - [x] I added a clear and descriptive title that summarizes this issue. - [x] ...
- **[#34085: `NotRequired` in TypedDict causes TypeError when converting to OpenAI function schema](https://github.com/langchain-ai/langchain/issues/34085)** [langchain] (🔴 open)
  > ### Checked other resources  - [x] This is a bug, not a usage question. - [x] I added a clear and descriptive title that summarizes this issue. - [x] ...
- **[#34797: Unable to use agent response format with ProviderStrategy and ChatOpenAI](https://github.com/langchain-ai/langchain/issues/34797)** [langchain] (🔴 open)
  > ### Checked other resources  - [x] This is a bug, not a usage question. - [x] I added a clear and descriptive title that summarizes this issue. - [x] ...
- **[#5054: TypeError: Type is not msgpack serializable: ToolMessage](https://github.com/langchain-ai/langgraph/issues/5054)** [langgraph] (🔴 open)
  > ``` 025-06-11 19:33:06 ERROR    /invoke exception! Type is not msgpack serializable: Send, repr: TypeError('Type is not msgpack serializable: Send') T...
- **[#6675: calling `model_dump()` on pydantic state variables drops `tool_calls` in AIMessage](https://github.com/langchain-ai/langgraph/issues/6675)** [langgraph] (🔴 open)
  > ### Checked other resources  - [x] This is a bug, not a usage question. For questions, please use the LangChain Forum (https://forum.langchain.com/). ...

## Diagnosis
**Root Cause:** The argument name 'config' is colliding with a reserved keyword or internal variable within LangChain or LangGraph's tool execution logic. This collision prevents the argument from being correctly passed to the tool function, resulting in the TypeError.
**Confidence:** High

**Supporting Evidence:**
- The error message 'TypeError: addConfig() missing 1 required positional argument: 'config'' indicates that the 'config' argument is not being passed correctly.
- Renaming the argument (e.g., to 'configf') resolves the issue, suggesting a name collision.
- The user's description explicitly mentions a suspicion of a keyword/name collision.
- The stack trace points to the tool execution logic within LangGraph's `prebuilt/tool_node.py` and LangChain's `core/tools/structured.py`, indicating the issue lies within the framework's argument handling.
- Similar issues have been observed where 'config' is a reserved word in LangGraph (see COMMON LANGCHAIN ECOSYSTEM ISSUES).

## Resolution Plan

### Step 1: Review the error message and stack trace carefully
*Why:* Understanding the exact error is the first step
*Expected result:* Identify the specific line or function causing the issue

### Step 2: Search for the error message online
*Why:* Others may have encountered and solved this issue
*Expected result:* Find relevant Stack Overflow posts or documentation

---
*If the issue persists after following these steps, please provide additional details.*
