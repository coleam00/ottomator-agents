from dotenv import load_dotenv
from httpx import AsyncClient
from datetime import datetime
import streamlit as st
import asyncio
import json
import os

from openai import AsyncOpenAI, OpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelTextResponse, UserPrompt

from web_search_agent import web_search_agent, Deps

load_dotenv()
llm = os.getenv('LLM_MODEL', 'gpt-4o')

# For now it appears Ollama doesn't support streaming with Pydantic AI so this is disabled
# If you want to use Ollama, you'll need to update this to be sync like web_search_agent.py

# client = AsyncOpenAI(
#     base_url = 'http://localhost:11434/v1',
#     api_key='ollama'
# )

# model = OpenAIModel('gpt-4o') if llm.lower().startswith("gpt") else OpenAIModel('qwen2.5-ottodev:32b', openai_client=client)

model = OpenAIModel('gpt-4o')

async def prompt_ai(messages):
    async with AsyncClient() as client:
        brave_api_key = os.getenv('BRAVE_API_KEY', None)
        deps = Deps(client=client, brave_api_key=brave_api_key)

        async with web_search_agent.run_stream(
            messages[-1].content, deps=deps, message_history=messages[:-1]
        ) as result:
            async for message in result.stream_text(delta=True):  
                yield message          

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    st.title("Pydantic AI Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message.role
        if role in ["user", "model-text-response"]:
            with st.chat_message("human" if role == "user" else "ai"):
                st.markdown(message.content)        

    # React to user input
    if prompt := st.chat_input("What would you like research today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(UserPrompt(content=prompt))

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in prompt_ai(st.session_state.messages):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)
        
        st.session_state.messages.append(ModelTextResponse(content=response_content))


if __name__ == "__main__":
    asyncio.run(main())