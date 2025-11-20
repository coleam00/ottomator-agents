from dotenv import load_dotenv
from datetime import datetime
from typing import List
import streamlit as st
import time
import json
import os

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = os.getenv('LLM_MODEL', 'gpt-4o')

mad_libs_prompt = """
You are an expert at creating funny and interesting Mad Libs. All of the Mad Libs you create have between 20 and 25 fill-in-the-blanks. 
Each fill-in-the-blank can be one of the following:

Noun
Verb
Adjective
Adverb 
Plural Noun 
Proper Noun 
Place 
Exclamation 
Part of the Body 
Occupation 
Emotion 
Verb Ending in -ing

Create a Mad Lib with the theme of {theme}. Output JSON in the format:

{{
    “text”: “Mad Lib text with exactly three underscores (like ___) to represent each fill-in-the-blank. Only include exactly 3 underscores for each blank. Don’t include the type of blank in this part of the JSON”,
    “blanks”: [Array of blanks where each element is the type of fill-in-the-blank such as Noun or Verb. List them in the order they are used in the Mad Lib text]
}}
"""

class MadLib(BaseModel):
    text: str = Field(description="Mad Lib text with three underscores (___) for each fill-in-the-blank")
    blanks: List[str] = Field(description="Array of blanks where each element is the type of fill-in-the-blank such as Noun or Verb listed in the order they are used in the Mad Lib text")

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

def main():
    st.title("AI Mad Libs!")

    # Initialize the chat history with the initial system message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content=f"Give me a theme and I'll make a Mad Lib for you!")
        ]

        # State will either be choosing-theme, choosing-words
        st.session_state.state = "choosing-theme"

    # Display chat messages from history each time the script is rerun when the UI state changes
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        if message_type in ["human", "ai"]:
            with st.chat_message(message_json["type"]):
                st.markdown(message_json["content"])        

    # Assign prompt to the user input if any is given, otherwise skip everything in this if statement
    if user_input := st.chat_input("Enter theme and fill-in-the-blanks here"):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)

        # Add the user message to chat history
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Get the current state of the chat which is either choosing-theme or choosing-words
        curr_state = st.session_state.state

        # If the user just selected a theme, generate the Madlib
        if curr_state == "choosing-theme":
            parser = JsonOutputParser(pydantic_object=MadLib)
            chatbot = ChatOpenAI(model=model)
            prompt = PromptTemplate(
                template=mad_libs_prompt,
                input_variables=["theme"]
            )

            # Create the chain with the prompt, model (GPT or Local), and MadLib output parser
            mad_lib_chain = prompt | chatbot | parser

            # Invoke the chain with the theme the user selected to generate the Madlib
            mad_lib_json = mad_lib_chain.invoke({"theme": user_input})    

            st.session_state.text = mad_lib_json["text"]
            st.session_state.blanks = mad_lib_json["blanks"]
            num_blanks = len(mad_lib_json["blanks"])
            st.session_state.count = num_blanks
            first_blank = mad_lib_json["blanks"][0]

            with st.chat_message("assistant"):
                response = st.write_stream(stream_text(f"Alright, let's get started!\n\n1 / {num_blanks}: {first_blank}"))
                st.session_state.messages.append(AIMessage(content=response))        

            st.session_state.state = "choosing-words"

        # If the user is picking words for a Madlib, add these message as a word and move to the next
        elif curr_state == "choosing-words":
            st.session_state.text = st.session_state.text.replace("___", f"**{user_input}**", 1)
            st.session_state.blanks = st.session_state.blanks[1:]
            blanks = st.session_state.blanks
            num_blanks = st.session_state.count

            # Still more blanks to fill in
            if len(blanks):
                next_blank = blanks[0]
                with st.chat_message("assistant"):
                    response = st.write_stream(stream_text(f"{num_blanks - len(blanks) + 1} / {num_blanks}: {next_blank}"))
                    st.session_state.messages.append(AIMessage(content=response))                     
            # No more blanks to fill in
            else:
                with st.chat_message("assistant"):
                    response = st.write_stream(stream_text(st.session_state.text))
                    st.session_state.messages.append(AIMessage(content=response))    

                st.session_state.state = "choosing-theme"


if __name__ == "__main__":
    main()