from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Annotated, Literal, Dict
from dotenv import load_dotenv
import streamlit as st
import json
import os

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace

from tools.asana_tools import available_asana_functions
from tools.google_drive_tools import available_drive_functions
from tools.vector_db_tools import available_vector_db_functions

load_dotenv()
model = os.getenv('LLM_MODEL', 'gpt-4o')
provider = os.getenv('LLM_PROVIDER', 'auto')

provider_mapping = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
    "ollama": ChatOllama,
    "llama": ChatGroq
}

model_mapping = {
    "gpt": ChatOpenAI,
    "claude": ChatAnthropic,
    "groq": ChatGroq,
    "llama": ChatGroq
}

# Support for HuggingFace with local models coming soon! This function isn't used yet.
@st.cache_resource
def get_local_model():
    return HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False
    )

    # If you want to run the model absolutely locally - VERY resource intense!
    # return HuggingFacePipeline.from_model_id(
    #     model_id=model,
    #     task="text-generation",
    #     pipeline_kwargs={
    #         "max_new_tokens": 1024,
    #         "top_k": 50,
    #         "temperature": 0.4
    #     },
    # )

available_functions = available_asana_functions | available_drive_functions | available_vector_db_functions
tools = [tool for _, tool in available_functions.items()]

if provider == "auto":
    for key, chatbot_class in model_mapping.items():
        if key in model.lower():
            chatbot = chatbot_class(model=model) if key != "huggingface" else chatbot_class(llm=get_local_model())
            break
else:
    for key, chatbot_class in provider_mapping.items():
        if key in provider.lower():
            chatbot = chatbot_class(model=model) if key != "huggingface" else chatbot_class(llm=get_local_model())
            break

chatbot_with_tools = chatbot.bind_tools(tools)

### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of chat messages.
    """
    messages: Annotated[list[AnyMessage], add_messages]

def call_model(state: GraphState, config: RunnableConfig) -> Dict[str, AnyMessage]:
    """
    Function that calls the model to generate a response.

    Args:
        state (GraphState): The current graph state

    Returns:
        dict: The updated state with a new AI message
    """
    print("---CALL MODEL---")

    messages = list(filter(
        lambda m: not isinstance(m, AIMessage) or hasattr(m, "response_metadata") and m.response_metadata, 
        state["messages"]
    ))

    # Invoke the chatbot with the binded tools
    response = chatbot_with_tools.invoke(messages, config)
    # print("Response from model:", response)

    # We return an object because this will get added to the existing list
    return {"messages": response}

def tool_node(state: GraphState) -> Dict[str, AnyMessage]:
    """
    Function that handles all tool calls.

    Args:
        state (GraphState): The current graph state

    Returns:
        dict: The updated state with tool messages
    """
    print("---TOOL NODE---")
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    outputs = []

    if last_message and last_message.tool_calls:
        for call in last_message.tool_calls:
            tool = available_functions.get(call['name'], None)

            if tool is None:
                raise Exception(f"Tool '{call['name']}' not found.")

            print(f"\n\nInvoking tool: {call['name']} with args {call['args']}")
            output = tool.invoke(call['args'])
            print(f"Result of invoking tool: {output}\n\n")

            outputs.append(ToolMessage(
                output if isinstance(output, str) else json.dumps(output), 
                tool_call_id=call['id']
            ))

    return {'messages': outputs}

def should_continue(state: GraphState) -> Literal["__end__", "tools"]:
    """
    Determine whether to continue or end the workflow based on if there are tool calls to make.

    Args:
        state (GraphState): The current graph state

    Returns:
        str: The next node to execute or END
    """
    print("---SHOULD CONTINUE---")
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # If there is no function call, then we finish
    if not last_message or not last_message.tool_calls:
        return END
    else:
        return "tools"

def get_runnable():
    workflow = StateGraph(GraphState)

    # Define the nodes and how they connect
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue
    )
    workflow.add_edge("tools", "agent")

    # Compile the LangGraph graph into a runnable
    memory = AsyncSqliteSaver.from_conn_string(":memory:")
    app = workflow.compile(checkpointer=memory)

    return app