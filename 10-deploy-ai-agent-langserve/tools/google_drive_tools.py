from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from langchain_core.tools import tool
import os
import io

SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]

def get_google_drive_service():
  """
  Gets the Google Drive credentials with the scope of full access to Drive files
  """
  creds = None
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials/credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  return build("drive", "v3", credentials=creds)

service = get_google_drive_service()

@tool
def search_file(query: str) -> list:
    """
    Searches for files in Google Drive based on a query string.
    
    Arguments:
    - query (str): The search query to find files. This requires a specific format for Google Drive:
    To search for files that have 'example' in the name - query should be: name contains 'example'
    To search for files that have 'example text' in the file text - query should be: fullText contains 'example text'
    
    Returns:
    - list: A list of dictionaries containing the file ID and name of the matched files.
    
    Example usage:
    search_file("name contains 'report'")
    """
    try:
        results = service.files().list(q=f"mimeType!='application/vnd.google-apps.folder' and {query}", spaces='drive', fields="files(id, name)").execute()
        return str(results.get('files', []))
    except Exception as e:
        return f"Failed to search Google Drive: {e}"

@tool
def download_file(file_id: str, file_name: str, mime_type: str = 'text/plain') -> str:
    """
    Downloads a Google Docs file (or similar) from Google Drive and saves it to a specified path.
    
    Arguments:
    - file_id (str): The unique ID of the file to be downloaded.
    - file_name (str): The name of the file (including the extension) to download it locally as.
    - mime_type (str, optional): The MIME type to export the file as. Defaults to 'text/plain'.
    
    Returns:
    - str: A message confirming the file has been downloaded to the specified path.
    
    Example usage:
    download_file("1aBcDeFgHiJkLmNoPqRsTuVwXyZ", "file.txt", "text/plain")
    """
    try:
        directory = "data"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        request = service.files().export_media(fileId=file_id, mimeType=mime_type)
        file_path = f"{directory}/{file_name}"
        with io.FileIO(file_path, 'wb') as file:
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        
        return f"File downloaded to {file_path}"
    except Exception as e:
        return f"Error downloading the file: {e}"


@tool
def upload_file(file_path: str, folder_id: str = None) -> str:
    """
    Uploads a file to a specific folder in Google Drive. If no folder ID is provided, it uploads to the root directory.
    
    Arguments:
    - file_path (str): The local path to the file that will be uploaded.
    - folder_id (str, optional): The ID of the Google Drive folder where the file will be uploaded. Defaults to None (uploads to root).
    
    Returns:
    - str: The ID of the uploaded file.
    
    Example usage:
    upload_file("/path/to/local/file.txt", "1aBcDeFgHiJkLmNoPqRsTuVwXyZ")
    """
    try:
        file_metadata = {'name': file_path.split("/")[-1]}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return f"File uploaded with ID: {file.get('id')}"
    except Exception as e:
        return f"Error uploading the file: {e}"

@tool
def delete_file(file_id: str) -> str:
    """
    Deletes a file from Google Drive based on its file ID.
    
    Arguments:
    - file_id (str): The unique ID of the file to be deleted.
    
    Returns:
    - str: A message confirming the deletion of the file.
    
    Example usage:
    delete_file("1aBcDeFgHiJkLmNoPqRsTuVwXyZ")
    """
    try:
        service.files().delete(fileId=file_id).execute()
        return f"File with ID {file_id} has been deleted."
    except Exception as e:
        return f"Error deleting the file: {e}"

@tool
def update_file(file_id: str, new_file_path: str) -> str:
    """
    Updates the contents of a file in Google Drive by replacing it with a new file.
    
    Arguments:
    - file_id (str): The unique ID of the file to be updated.
    - new_file_path (str): The local path to the new file that will replace the existing file.
    
    Returns:
    - str: A message confirming the file has been updated.
    
    Example usage:
    update_file("1aBcDeFgHiJkLmNoPqRsTuVwXyZ", "/path/to/new/file.txt")
    """
    try:
        media = MediaFileUpload(new_file_path, resumable=True)
        updated_file = service.files().update(fileId=file_id, media_body=media).execute()
        return f"File with ID {file_id} has been updated."
    except Exception as e:
        return f"Error updating the file: {e}"

@tool
def search_folder(query: str) -> list:
    """
    Searches for folders in Google Drive based on a query string.
    
    Arguments:
    - query (str): The search query to find folders - just the name or part of the name of folder(s) to search for.
    
    Returns:
    - list: A list of dictionaries containing the folder ID and name of the matched folders.
    
    Example usage:
    search_folder("name contains 'meeting_notes'")
    """
    try:
        results = service.files().list(q=f"mimeType='application/vnd.google-apps.folder' and name contains '{query}'",
                                    spaces='drive', fields="files(id, name)").execute()
        return str(results.get('files', []))
    except Exception as e:
        return f"Error searching folders: {e}"

@tool
def create_folder(folder_name: str, parent_folder_id: str = None) -> str:
    """
    Creates a folder in Google Drive. If a parent folder ID is provided, the folder is created inside that folder.
    
    Arguments:
    - folder_name (str): The name of the folder to be created.
    - parent_folder_id (str, optional): The ID of the parent folder where the new folder will be created. Defaults to None (creates in root).
    
    Returns:
    - str: The ID of the created folder.
    
    Example usage:
    create_folder("New Meeting Folder", "1aBcDeFgHiJkLmNoPqRsTuVwXyZ")
    """
    try:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_folder_id:
            file_metadata['parents'] = [parent_folder_id]
        folder = service.files().create(body=file_metadata, fields='id').execute()
        return f"Folder created with ID: {folder.get('id')}"
    except Exception as e:
        return f"Error creating the folder: {e}"

@tool
def delete_folder(folder_id: str) -> str:
    """
    Deletes a folder from Google Drive based on its folder ID.
    
    Arguments:
    - folder_id (str): The unique ID of the folder to be deleted.
    
    Returns:
    - str: A message confirming the deletion of the folder.
    
    Example usage:
    delete_folder("1aBcDeFgHiJkLmNoPqRsTuVwXyZ")
    """
    try:
        service.files().delete(fileId=folder_id).execute()
        return f"Folder with ID {folder_id} has been deleted."
    except Exception as e:
        return f"Error deleting the folder: {e}"

@tool
def create_text_file(content: str, file_name: str) -> str:
    """
    Creates a text file with the given content + file name and returns the file path.
    
    Arguments:
    - content (str): The text content to be written to the file.
    - file_name (str): The name of the file to be created (including the file extension, typically .txt).
    
    Returns:
    - str: The path to the created text file.
    
    Example usage:
    create_text_file("Hello, world!", "example.txt")
    """
    try:
        directory = "data"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        file_path = f"{directory}/{file_name}"
        with open(file_path, "w") as file:
            file.write(content)
        return file_path
    except Exception as e:
        return f"Error creating the text file: {e}"

# Maps the function names to the actual function object in the script
# This mapping will also be used to create the list of tools to bind to the agent
available_drive_functions = {
  "search_file": search_file,
  "download_file": download_file,
  "upload_file": upload_file,
  "delete_file": delete_file,
  "update_file": update_file,
  "search_folder": search_folder,
  "create_folder": create_folder,
  "delete_folder": delete_folder,
  "create_text_file": create_text_file
}
