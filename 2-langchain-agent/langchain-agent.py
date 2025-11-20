import asana
from asana.rest import ApiException
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import json
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

model = os.getenv('LLM_MODEL', 'gpt-4o')

configuration = asana.Configuration()
configuration.access_token = os.getenv('ASANA_ACCESS_TOKEN', '')
api_client = asana.ApiClient(configuration)

tasks_api_instance = asana.TasksApi(api_client)

@tool
def create_asana_task(task_name, due_on="today"):
    """
    Creates a task in Asana given the name of the task and when it is due

    Example call:

    create_asana_task("Test Task", "2024-06-24")
    Args:
        task_name (str): The name of the task in Asana
        due_on (str): The date the task is due in the format YYYY-MM-DD. If not given, the current day is used
    Returns:
        str: The API response of adding the task to Asana or an error message if the API call threw an error
    """
    if due_on == "today":
        due_on = str(datetime.now().date())

    task_body = {
        "data": {
            "name": task_name,
            "due_on": due_on,
            "projects": [os.getenv("ASANA_PROJECT_ID", "")]
        }
    }

    try:
        api_response = tasks_api_instance.create_task(task_body, {})
        return json.dumps(api_response, indent=2)
    except ApiException as e:
        return f"Exception when calling TasksApi->create_task: {e}"  

def prompt_ai(messages, nested_calls=0):
    if nested_calls > 5:
        raise "AI is tool calling too much!"

    # First, prompt the AI with the latest user message
    tools = [create_asana_task]
    asana_chatbot = ChatOpenAI(model=model) if "gpt" in model.lower() else ChatAnthropic(model=model)
    asana_chatbot_with_tools = asana_chatbot.bind_tools(tools)

    ai_response = asana_chatbot_with_tools.invoke(messages)
    print(ai_response)
    print(type(ai_response))
    tool_calls = len(ai_response.tool_calls) > 0

    # Second, see if the AI decided it needs to invoke a tool
    if tool_calls:
        # If the AI decided to invoke a tool, invoke it
        available_functions = {
            "create_asana_task": create_asana_task
        }

        # Add the tool request to the list of messages so the AI knows later it invoked the tool
        messages.append(ai_response)


        # Next, for each tool the AI wanted to call, call it and add the tool result to the list of messages
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"].lower()
            selected_tool = available_functions[tool_name]
            tool_output = selected_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))                

        # Call the AI again so it can produce a response with the result of calling the tool(s)
        ai_response = prompt_ai(messages, nested_calls + 1)

    return ai_response

def main():
    messages = [
        SystemMessage(content=f"You are a personal assistant who helps manage tasks in Asana. The current date is: {datetime.now().date()}")
    ]

    while True:
        user_input = input("Chat with AI (q to quit): ").strip()
        
        if user_input == 'q':
            break  

        messages.append(HumanMessage(content=user_input))
        ai_response = prompt_ai(messages)

        print(ai_response.content)
        messages.append(ai_response)


if __name__ == "__main__":
    main()