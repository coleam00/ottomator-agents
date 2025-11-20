import asana
from asana.rest import ApiException
from dotenv import load_dotenv
from datetime import datetime
from typing import List
import streamlit as st
import uuid
import json
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

model = os.getenv('LLM_MODEL', 'o1-mini')
show_thoughts = os.getenv('SHOW_THOUGHTS', 'true').lower() in ["true", "yes", "1"]

configuration = asana.Configuration()
configuration.access_token = os.getenv('ASANA_ACCESS_TOKEN', '')
api_client = asana.ApiClient(configuration)

# create an instance of the different Asana API classes
projects_api_instance = asana.ProjectsApi(api_client)
tasks_api_instance = asana.TasksApi(api_client)

workspace_gid = os.getenv("ASANA_WORKPLACE_ID", "")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ AI Agent Tool Functions ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@tool
def create_asana_task(task_name, project_gid, due_on="today"):
    """
    Creates a task in Asana given the name of the task and when it is due

    Example call:

    create_asana_task("Test Task", "2024-06-24")
    Args:
        task_name (str): The name of the task in Asana
        project_gid (str): The ID of the project to add the task to
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
            "projects": [project_gid]
        }
    }

    try:
        api_response = tasks_api_instance.create_task(task_body, {})
        return json.dumps(api_response, indent=2)
    except ApiException as e:
        return f"Exception when calling TasksApi->create_task: {e}"  

@tool
def get_asana_projects():
    """
    Gets all of the projects in the user's Asana workspace

    Returns:
        str: The API response from getting the projects or an error message if the projects couldn't be fetched.
        The API response is an array of project objects, where each project object looks like:
        {'gid': '1207789085525921', 'name': 'Project Name', 'resource_type': 'project'}
    """    
    opts = {
        'limit': 50, # int | Results per page. The number of objects to return per page. The value must be between 1 and 100.
        'workspace': workspace_gid, # str | The workspace or organization to filter projects on.
        'archived': False # bool | Only return projects whose `archived` field takes on the value of this parameter.
    }

    try:
        api_response = projects_api_instance.get_projects(opts)
        return json.dumps(list(api_response), indent=2)
    except ApiException as e:
        return "Exception when calling ProjectsApi->create_project: %s\n" % e

@tool
def create_asana_project(project_name, due_on=None):
    """
    Creates a project in Asana given the name of the project and optionally when it is due

    Example call:

    create_asana_project("Test Project", "2024-06-24")
    Args:
        project_name (str): The name of the project in Asana
        due_on (str): The date the project is due in the format YYYY-MM-DD. If not supplied, the project is not given a due date
    Returns:
        str: The API response of adding the project to Asana or an error message if the API call threw an error
    """    
    body = {
        "data": {
            "name": project_name, "due_on": due_on, "workspace": workspace_gid
        }
    } # dict | The project to create.

    try:
        # Create a project
        api_response = projects_api_instance.create_project(body, {})
        return json.dumps(api_response, indent=2)
    except ApiException as e:
        return "Exception when calling ProjectsApi->create_project: %s\n" % e  

@tool
def get_asana_tasks(project_gid):
    """
    Gets all the Asana tasks in a project

    Example call:

    get_asana_tasks("1207789085525921")
    Args:
        project_gid (str): The ID of the project in Asana to fetch the tasks for
    Returns:
        str: The API response from fetching the tasks for the project in Asana or an error message if the API call threw an error
        The API response is an array of tasks objects where each task object is in the format:
        {'gid': '1207780961742158', 'created_at': '2024-07-11T16:25:46.380Z', 'due_on': None or date in format "YYYY-MM-DD", 'name': 'Test Task'}
    """        
    opts = {
        'limit': 50, # int | Results per page. The number of objects to return per page. The value must be between 1 and 100.
        'project': project_gid, # str | The project to filter tasks on.
        'opt_fields': "created_at,name,due_on", # list[str] | This endpoint returns a compact resource, which excludes some properties by default. To include those optional properties, set this query parameter to a comma-separated list of the properties you wish to include.
    }

    try:
        # Get multiple tasks
        api_response = tasks_api_instance.get_tasks(opts)
        return json.dumps(list(api_response), indent=2)
    except ApiException as e:
        return "Exception when calling TasksApi->get_tasks: %s\n" % e

@tool
def update_asana_task(task_gid, data):
    """
    Updates a task in Asana by updating one or both of completed and/or the due date

    Example call:

    update_asana_task("1207780961742158", {"completed": True, "due_on": "2024-07-13"})
    Args:
        task_gid (str): The ID of the task to update
        data (dict): A dictionary with either one or both of the keys 'completed' and/or 'due_on'
                    If given, completed needs to be either True or False.
                    If given, the due date needs to be in the format 'YYYY-MM-DD'.
    Returns:
        str: The API response of updating the task or an error message if the API call threw an error
    """      
    # Data: {"completed": True or False, "due_on": "YYYY-MM-DD"}
    body = {"data": data} # dict | The task to update.

    try:
        # Update a task
        api_response = tasks_api_instance.update_task(body, task_gid, {})
        return json.dumps(api_response, indent=2)
    except ApiException as e:
        return "Exception when calling TasksApi->update_task: %s\n" % e

@tool
def delete_task(task_gid):
    """
    Deletes a task in Asana

    Example call:

    delete_task("1207780961742158")
    Args:
        task_gid (str): The ID of the task to delete
    Returns:
        str: The API response of deleting the task or an error message if the API call threw an error
    """        
    try:
        # Delete a task
        api_response = tasks_api_instance.delete_task(task_gid)
        return json.dumps(api_response, indent=2)
    except ApiException as e:
        return "Exception when calling TasksApi->delete_task: %s\n" % e   

# Maps the function names to the actual function object in the script
# This mapping will also be used to create the list of tools to bind to the agent
available_tools = {
    "create_asana_task": create_asana_task,
    "get_asana_projects": get_asana_projects,
    "create_asana_project": create_asana_project,
    "get_asana_tasks": get_asana_tasks,
    "update_asana_task": update_asana_task,
    "delete_task": delete_task
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~ Tool Prompt Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tool_descriptions = [f"{name}:\n{func.__doc__}\n\n" for name, func in available_tools.items()]

class ToolCall(BaseModel):
    name: str = Field(description="Name of the function to run")
    args: dict = Field(description="Arguments for the function call (empty dictionary if no arguments are needed for the tool call)")    

class ToolCallOrResponse(BaseModel):
    tool_calls: List[ToolCall] = Field(description="List of tool calls, empty array if you don't need to invoke a tool")
    content: str = Field(description="Response to the user if a tool doesn't need to be invoked")

tool_text = f"""
You always respond with a JSON object that has two required keys.

tool_calls: List[ToolCall] = Field(description="List of tool calls, empty array if you don't need to invoke a tool")
content: str = Field(description="Response to the user if a tool doesn't need to be invoked")

Here is the type for ToolCall (object with two keys):
    name: str = Field(description="Name of the function to run (NA if you don't need to invoke a tool)")
    args: dict = Field(description="Arguments for the function call (empty dictionary if you don't need to invoke a tool or if no arguments are needed for the tool call)")

Don't start your answers with "Here is the JSON response", just give the JSON.

The tools you have access to are:

{"".join(tool_descriptions)}

Any message that starts with "Thought:" is you thinking to yourself. This isn't told to the user so you still need to communicate what you did with them.
Don't repeat an action. If a thought tells you that you already took an action for a user, don't do it again.
"""    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~ AI Prompting Function ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def add_thought(thought):
    """
    Important function that adds LLM "thoughts" to the conversation
    that can optionally be show to the user. This includes things like
    results of tool calls, the LLM correcting itself, etc.
    """
    st.session_state.messages.append(AIMessage(content=thought))     

    # Show thoughts determined by .env variable SHOW_THOUGHTS
    if show_thoughts:
        with st.chat_message("assistant"):
            st.markdown(thought)       

def prompt_ai(nested_calls=0, invoked_tools=[]):
    if nested_calls > 10:
        raise Exception("Failsafe - AI is failing too much!")

    # First, prompt the AI with the latest user message
    parser = JsonOutputParser(pydantic_object=ToolCallOrResponse)
    asana_chatbot = ChatOpenAI(model=model, temperature=1) | parser

    try:
        ai_response = asana_chatbot.invoke(st.session_state.messages)
    except Exception as e:
        print(e)
        return prompt_ai(nested_calls + 1)
    print(ai_response)

    # Second, see if the AI decided it needs to invoke a tool
    has_tool_calls = len(ai_response["tool_calls"]) > 0
    if has_tool_calls:
        # Next, for each tool the AI wanted to call, call it and add the tool result to the list of messages as a "thought" for the LLM
        for tool_call in ai_response["tool_calls"]:
            if str(tool_call) not in invoked_tools:
                tool_name = tool_call["name"].lower()
                selected_tool = available_tools[tool_name]

                # Invoke the tool and add the response as a thought
                try:
                    tool_output = selected_tool.invoke(tool_call["args"])
                except Exception as e:
                    # AI gave bad arguments for the function, so add that as a thought and have the LLM correct itself
                    add_thought(f"Thought: - I called {tool_name} with args {tool_call['args']} but my arguments were wrong so I got this error: {e}.")   
                    return prompt_ai(nested_calls + 1, invoked_tools)

                print(tool_output)

                # Add a thought so the LLM knows the result of invoking the tool
                add_thought(f"Thought: - I called {tool_name} with args {tool_call['args']} and got back: {tool_output}.")    

                # Add to the list of tool calls so this app can prevent the LLM from repeating itself
                invoked_tools.append(str(tool_call))  
            else:
                # In this case the LLM already tried to make the exact same tool call. So add a thought for that so it doesn't loop.
                add_thought(f"Thought: - I already called {tool_call['name']} with args {tool_call['args']} and got a response. I need to respond to the user now and not make another tool call.")                            

        # Prompt the AI again now that the result of calling the tool(s) has been added to the chat history
        return prompt_ai(nested_calls + 1, invoked_tools)

    return ai_response

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    st.title("o1 Agent Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            HumanMessage(content=f"You are a personal assistant who helps manage tasks in Asana. The current date is: {datetime.now().date()}.\n{tool_text}")
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        message_content = message_json["content"]
        if message_type in ["human", "ai"] and (not message_content.startswith("Thought:") or show_thoughts):
            with st.chat_message(message_type):
                st.markdown(message_content)        

    # React to user input
    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        ai_response = prompt_ai()
        with st.chat_message("assistant"):
            st.markdown(ai_response['content'])
        
        st.session_state.messages.append(AIMessage(content=ai_response['content']))


if __name__ == "__main__":
    main()