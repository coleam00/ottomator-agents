from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

load_dotenv()

model = os.getenv('LLM_MODEL', 'gpt-4o')

def main():
    st.title("Streamlit Chatbot")

    # Initialize the chat history with the initial system message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"The current date is: {datetime.now().date()}")
        ]

    # Display chat messages from history each time the script is rerun when the UI state changes
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        with st.chat_message(message_json["type"]):
            st.markdown(message_json["content"])        

    # Assign prompt to the user input if any is given, otherwise skip everything in this if statement
    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add the user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display the chatbot's response in chat message container
        with st.chat_message("assistant"):
            chatbot = ChatOpenAI(model=model)
            stream = chatbot.stream(st.session_state.messages)
            response = st.write_stream(stream)

        st.session_state.messages.append(AIMessage(content=response))


if __name__ == "__main__":
    main()