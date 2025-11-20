from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import asyncio
import json
import os

from tools import available_functions

load_dotenv()
model = os.getenv('LLM_MODEL', 'gpt-4o')

system_message = f"""
You are a personal assistant who helps with research, managing Google Drive, and managing Slack. 
You never give IDs to the user since those are just for you to keep track of.
The link to any Google Doc is: https://docs.google.com/document/d/[document ID]
The current date is: {datetime.now().date()}
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~ AI Prompting Function ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_chunk_text(chunk):
    response_content = ""
    chunk_content = chunk.content
    if isinstance(chunk_content, str):
        response_content += chunk_content
    elif isinstance(chunk_content, list):
        for chunk_text in chunk_content:
            if "text" in chunk_text:
                response_content += chunk_text["text"]

    return response_content  

def prompt_ai(messages):
    # First, prompt the AI with the latest user message
    tools = [tool for _, tool in available_functions.items()]
    n8n_chatbot = ChatOpenAI(model=model) if "gpt" in model.lower() else ChatAnthropic(model=model)
    n8n_chatbot_with_tools = n8n_chatbot.bind_tools(tools)

    stream = n8n_chatbot_with_tools.stream(messages)
    first = True
    for chunk in stream:
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk

        yield get_chunk_text(chunk)

    has_tool_calls = len(gathered.tool_calls) > 0

    # Second, see if the AI decided it needs to invoke a tool
    if has_tool_calls:
        # Add the tool request to the list of messages so the AI knows later it invoked the tool
        messages.append(gathered)

        # If the AI decided to invoke a tool, invoke it
        # For each tool the AI wanted to call, call it and add the tool result to the list of messages
        for tool_call in gathered.tool_calls:
            tool_name = tool_call["name"].lower()
            selected_tool = available_functions[tool_name]
            print(f"\nInvoking tool: {tool_call['name']} with args {tool_call['args']}")
            tool_output = selected_tool.invoke(tool_call["args"])
            print(f"Result of invoking tool: {tool_output}\n")
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))                

        # Call the AI again so it can produce a response with the result of calling the tool(s)
        additional_stream = prompt_ai(messages)
        for additional_chunk in additional_stream:
            yield additional_chunk

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    st.title("n8n LangChain Agent")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=system_message)
        ]    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])        

    # React to user input
    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            stream = prompt_ai(st.session_state.messages)
            response = st.write_stream(stream)
        
            st.session_state.messages.append(AIMessage(content=response))


if __name__ == "__main__":
    asyncio.run(main())