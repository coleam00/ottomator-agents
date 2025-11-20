from dotenv import load_dotenv
import streamlit as st
import requests
import uuid
import json
import os

from langchain_core.messages import AIMessage, HumanMessage

"""
This Python script is an example of how to use Streamlit with
an n8n AI Agent as a webhook (API endpoint). This code pretty much just
defines a Streamlit UI that interacts with the n8n AI Agent for
each user message and displays the AI response from n8n back to the
UI just like other AI Agents in this masterclass. All chat history
and tool calling is managed within the n8n workflow.
"""

load_dotenv()

webhook_url = os.getenv('WEBHOOK_URL')
webhook_auth = os.getenv('WEBHOOK_AUTH')

@st.cache_resource
def get_session_id():
    return str(uuid.uuid4())

session_id = get_session_id()

def prompt_ai(user_input):
    payload = {
        "chatInput": user_input,
        "sessionId": session_id
    }

    headers = {
        "Authorization": f"Bearer {webhook_auth}",
        "Content-Type": "application/json"
    }

    response = requests.post(webhook_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    st.title("N8N Asana Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []    

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
            response = prompt_ai(prompt)
            st.markdown(response["output"])
        
        st.session_state.messages.append(AIMessage(content=response["output"]))


if __name__ == "__main__":
    main()