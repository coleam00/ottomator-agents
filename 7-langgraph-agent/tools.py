import asana
from asana.rest import ApiException
from dotenv import load_dotenv
import json
import os

from langchain_core.tools import tool

load_dotenv()

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
available_functions = {
    "create_asana_task": create_asana_task,
    "get_asana_projects": get_asana_projects,
    "create_asana_project": create_asana_project,
    "get_asana_tasks": get_asana_tasks,
    "update_asana_task": update_asana_task,
    "delete_task": delete_task
}  