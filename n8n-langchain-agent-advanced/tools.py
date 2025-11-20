from dotenv import load_dotenv
import requests
import json
import os

from langchain_core.tools import tool

load_dotenv()

N8N_BEARER_TOKEN = os.environ["N8N_BEARER_TOKEN"]
SUMMARIZE_SLACK_CONVERSATION_WEBHOOK = os.environ["SUMMARIZE_SLACK_CONVERSATION_WEBHOOK"]
SEND_SLACK_MESSAGE_WEBHOOK = os.environ["SEND_SLACK_MESSAGE_WEBHOOK"]
UPLOAD_GOOGLE_DOC_WEBHOOK = os.environ["UPLOAD_GOOGLE_DOC_WEBHOOK"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~ Helper Function for Invoking n8n Webhooks ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def invoke_n8n_webhook(method, url, function_name, payload=None):
    """
    Helper function to make a GET or POST request.

    Args:
        method (str): HTTP method ('GET' or 'POST')
        url (str): The API endpoint
        function_name (str): The name of the tool the AI agent invoked
        payload (dict, optional): The payload for POST requests

    Returns:
        str: The API response in JSON format or an error message
    """
    headers = {
        "Authorization": f"Bearer {N8N_BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=payload)
        else:
            return f"Unsupported method: {method}"

        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Exception when calling {function_name}: {e}" 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~ n8n AI Agent Tool Functions ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@tool
def summarize_slack_conversation():
    """
    Gets the latest messages in a Slack channel and summarizes the conversation

    Example call:

    summarize_slack_conversation()
    Args:
        None
    Returns:
        str: The API response with the Slack conversation summary or an error if there was an issue
    """
    return invoke_n8n_webhook(
        "GET",
        SUMMARIZE_SLACK_CONVERSATION_WEBHOOK,
        "summarize_slack_conversation"
    )  

@tool
def send_slack_message(message):
    """
    Sends a message in a Slack channel

    Example call:

    send_slack_message("Greetings!")
    Args:
        message (str): The message to send in the Slack channel
    Returns:
        str: The API response with the result of sending the Slack message or an error if there was an issue
    """
    return invoke_n8n_webhook(
        "POST",
        SEND_SLACK_MESSAGE_WEBHOOK,
        "send_slack_message",
        {"message": message}
    )  

@tool
def create_google_doc(document_title, document_text):
    """
    Creates a Google Doc in Google Drive with the text specified.

    Example call:

    create_google_doc("9/20 Meeting Notes", "Meeting notes for 9/20...")
    Args:
        document_title (str): The name of the Google Doc
        document_text (str): The text to put in the new Google Doc
    Returns:
        str: The API response with the result of creating the Google Doc or an error if there was an issue
    """
    return invoke_n8n_webhook(
        "POST",
        UPLOAD_GOOGLE_DOC_WEBHOOK,
        "create_google_doc",
        {"document_title": document_title, "document_text": document_text}
    )  

# Maps the function names to the actual function object in the script
# This mapping will also be used to create the list of tools to bind to the agent
available_functions = {
    "summarize_slack_conversation": summarize_slack_conversation,
    "send_slack_message": send_slack_message,
    "create_google_doc": create_google_doc
}  