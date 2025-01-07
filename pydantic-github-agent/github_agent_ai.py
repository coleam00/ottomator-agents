# Changes to github_agent_ai.py

from __future__ import annotations

import asyncio
import os
import datetime  # New import for timestamp handling
from dataclasses import dataclass
from typing import Any, List, Dict, Optional  # Added Optional type
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import shutil
import time
import re
import json

import httpx
from git import Repo
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from devtools import debug

load_dotenv()



# Initialize the model first
llm = os.getenv('LLM_MODEL')  # Default to gpt-4 if not specified
model = OpenAIModel(
    llm,
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPEN_AI_BASE_URL',)  # Optional base URL
)
# Enhanced Deps class with additional configuration
@dataclass
class Deps:
    client: httpx.AsyncClient
    github_token: str | None = None
    rate_limit_delay: float = 1.0
    max_retries: int = 2
    cache_duration: int = 300

    async def handle_rate_limit(self, response: httpx.Response) -> None:
    # Added method for rate limit handling
        if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
            remaining = int(response.headers['X-RateLimit-Remaining'])
            if remaining == 0:
                reset_time = int(response.headers['X-RateLimit-Reset'])
                current_time = int(time.time())
                sleep_time = reset_time - current_time + 1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

    # Added method for making GitHub API requests with retries and rate limiting
    async def github_request(self, url: str, method: str = "GET", **kwargs) -> httpx.Response:
        headers = kwargs.get('headers', {})
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        kwargs['headers'] = headers

        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                await self.handle_rate_limit(response)
                if response.status_code == 200:
                    return response
                if response.status_code != 429:  # If not rate limited, break
                    break
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except httpx.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)
        return response

# Enhanced system prompt with more detailed capabilities
system_prompt = """
You are a coding expert with access to GitHub to help the user manage their repository and get information from it.

You should engage in natural conversation with the user and autonomously use the GitHub tools when needed.
You don't need to ask permission before using tools - just use them when appropriate to answer the user's questions.

When you use a tool, incorporate its results naturally into your conversation rather than just showing raw output.
Make your responses friendly and informative.

Available tools:
- get_repo_info: Get basic repository information
- get_repo_structure: Get the file/directory structure
- get_file_content: Get contents of a specific file
- get_repo_issues: Get repository issues
- get_repo_pull_requests: Get pull requests
- get_repo_contributors: Get contributor information  
- get_repo_releases: Get release information
- search_code: Search code in the repository
- get_commit_history: Get recent commits
- analyze_code_frequency: Get code addition/deletion stats

Remember to handle errors gracefully and provide helpful responses even if a tool call fails.
"""

# Modified agent configuration with enhanced settings

github_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=Deps,
    retries=2
)


# Modified get_repo_info with enhanced error handling and caching
@github_agent.tool
async def get_repo_info(ctx: RunContext[Deps], github_url: str) -> str:
    """Get repository information including size and description using GitHub API.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Repository information as a formatted string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    
    response = await ctx.deps.github_request(
        f'https://api.github.com/repos/{owner}/{repo}'
    )
    
    if response.status_code != 200:
        return f"Failed to get repository info: {response.text}"
    
    data = response.json()
    size_mb = data['size'] / 1024
    
    # Enhanced response format with more details
    return (
        f"Repository: {data['full_name']}\n"
        f"Description: {data['description']}\n"
        f"Size: {size_mb:.1f}MB\n"
        f"Stars: {data['stargazers_count']}\n"
        f"Forks: {data['forks_count']}\n"
        f"Open Issues: {data['open_issues_count']}\n"
        f"Language: {data['language']}\n"
        f"Created: {data['created_at']}\n"
        f"Last Updated: {data['updated_at']}\n"
        f"License: {data.get('license', {}).get('name', 'Not specified')}\n"
        f"Topics: {', '.join(data.get('topics', ['None']))}\n"
        f"Visibility: {data['visibility']}"
    )

# Modified get_repo_structure with enhanced filtering and organization
@github_agent.tool
async def get_repo_structure(ctx: RunContext[Deps], github_url: str, path: str = "") -> str:
    """Get the directory structure of a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        path: Optional path within the repository to get structure for.

    Returns:
        str: Directory structure as a formatted string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    
    # Try main branch first, then master
    branches = ['main', 'master']
    tree_data = None
    
    for branch in branches:
        response = await ctx.deps.github_request(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}',
            params={'recursive': '1'}
        )
        
        if response.status_code == 200:
            tree_data = response.json()
            break
    
    if not tree_data:
        return "Failed to get repository structure"
    
    # Enhanced filtering and organization
    excluded_patterns = [
        '.git/', 
        'node_modules/', 
        '__pycache__/',
        '.pytest_cache/',
        '.venv/',
        'dist/',
        'build/'
    ]
    
    def should_include(path: str) -> bool:
        return not any(pattern in path for pattern in excluded_patterns)
    
    # Organize by file type
    structure = {
        'Directories': [],
        'Python Files': [],
        'JavaScript Files': [],
        'Documentation': [],
        'Configuration': [],
        'Other Files': []
    }
    
    for item in tree_data['tree']:
        if not should_include(item['path']):
            continue
            
        if path and not item['path'].startswith(path):
            continue
            
        prefix = '📁 ' if item['type'] == 'tree' else '📄 '
        entry = f"{prefix}{item['path']}"
        
        if item['type'] == 'tree':
            structure['Directories'].append(entry)
        elif item['path'].endswith('.py'):
            structure['Python Files'].append(entry)
        elif item['path'].endswith('.js') or item['path'].endswith('.ts'):
            structure['JavaScript Files'].append(entry)
        elif item['path'].endswith(('.md', '.rst', '.txt')):
            structure['Documentation'].append(entry)
        elif item['path'].endswith(('.json', '.yaml', '.yml', '.toml', '.ini')):
            structure['Configuration'].append(entry)
        else:
            structure['Other Files'].append(entry)
    
    # Format output
    output = []
    for category, items in structure.items():
        if items:
            output.append(f"\n{category}:")
            output.extend(sorted(items))
    
    return "\n".join(output)

# Modified get_file_content with enhanced handling and formatting
@github_agent.tool
async def get_file_content(
    ctx: RunContext[Deps], 
    github_url: str, 
    file_path: str,
    branch: str = "main"
) -> str:
    """Get the content of a specific file from the GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        file_path: Path to the file within the repository.
        branch: Branch to get the file from (default: main)

    Returns:
        str: File content as a string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    
    # Try specified branch first, then fall back to master
    branches = [branch, 'master'] if branch == 'main' else [branch]
    
    for current_branch in branches:
        response = await ctx.deps.github_request(
            f'https://raw.githubusercontent.com/{owner}/{repo}/{current_branch}/{file_path}'
        )
        
        if response.status_code == 200:
            content = response.text
            
            # Add syntax highlighting hints based on file extension
            ext = Path(file_path).suffix.lower()
            if ext in ['.py', '.js', '.java', '.cpp', '.cs', '.php', '.rb', '.go', '.rs']:
                return f"```{ext[1:]}\n{content}\n```"
            elif ext in ['.md', '.markdown']:
                return f"```markdown\n{content}\n```"
            elif ext in ['.json', '.yaml', '.yml', '.toml']:
                return f"```{ext[1:]}\n{content}\n```"
            else:
                return content
    
    return f"Failed to get file content: File not found in specified branches"



@github_agent.tool
async def get_repo_issues(ctx: RunContext[Deps], github_url: str, state: str = "all") -> str:
    """Get repository issues using GitHub API.
    
    Args:
        ctx: The context
        github_url: The GitHub repository URL
        state: The state of issues to retrieve (open/closed/all)
    
    Returns:
        str: Issues information as a formatted string
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/issues?state={state}',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get repository issues: {response.text}"
    
    issues = response.json()
    formatted_issues = []
    
    for issue in issues:
        formatted_issues.append(
            f"#{issue['number']} - {issue['title']}\n"
            f"State: {issue['state']}\n"
            f"Created: {issue['created_at']}\n"
            f"Comments: {issue['comments']}\n"
        )
    
    return "\n".join(formatted_issues)

@github_agent.tool
async def get_repo_pull_requests(ctx: RunContext[Deps], github_url: str, state: str = "all") -> str:
    """Get repository pull requests using GitHub API.
    
    Args:
        ctx: The context
        github_url: The GitHub repository URL
        state: The state of PRs to retrieve (open/closed/all)
    
    Returns:
        str: Pull requests information as a formatted string
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/pulls?state={state}',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get repository pull requests: {response.text}"
    
    prs = response.json()
    formatted_prs = []
    
    for pr in prs:
        formatted_prs.append(
            f"#{pr['number']} - {pr['title']}\n"
            f"State: {pr['state']}\n"
            f"Created: {pr['created_at']}\n"
            f"User: {pr['user']['login']}\n"
            f"Comments: {pr['comments']}\n"
        )
    
    return "\n".join(formatted_prs)

@github_agent.tool
async def get_repo_contributors(ctx: RunContext[Deps], github_url: str) -> str:
    """Get repository contributors using GitHub API.
    
    Args:
        ctx: The context
        github_url: The GitHub repository URL
    
    Returns:
        str: Contributors information as a formatted string
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/contributors',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get repository contributors: {response.text}"
    
    contributors = response.json()
    formatted_contributors = []
    
    for contributor in contributors:
        formatted_contributors.append(
            f"Username: {contributor['login']}\n"
            f"Contributions: {contributor['contributions']}\n"
            f"Profile: {contributor['html_url']}\n"
        )
    
    return "\n".join(formatted_contributors)

@github_agent.tool
async def get_repo_releases(ctx: RunContext[Deps], github_url: str) -> str:
    """Get repository releases using GitHub API.
    
    Args:
        ctx: The context
        github_url: The GitHub repository URL
    
    Returns:
        str: Releases information as a formatted string
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/releases',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get repository releases: {response.text}"
    
    releases = response.json()
    formatted_releases = []
    
    for release in releases:
        formatted_releases.append(
            f"Tag: {release['tag_name']}\n"
            f"Name: {release['name']}\n"
            f"Published: {release['published_at']}\n"
            f"Assets: {len(release['assets'])}\n"
            f"URL: {release['html_url']}\n"
        )
    
    return "\n".join(formatted_releases)

@github_agent.tool
async def search_code(ctx: RunContext[Deps], github_url: str, query: str) -> str:
    """Search code in the repository using GitHub API.
    
    Args:
        ctx: The context
        github_url: The GitHub repository URL
        query: The search query
    
    Returns:
        str: Search results as a formatted string
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    # Format query to search in specific repository
    repo_query = f"repo:{owner}/{repo} {query}"
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/search/code',
        params={'q': repo_query},
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to search repository code: {response.text}"
    
    results = response.json()
    formatted_results = []
    
    for item in results['items'][:5]:  # Limiting to top 5 results
        formatted_results.append(
            f"File: {item['path']}\n"
            f"URL: {item['html_url']}\n"
            f"Repository: {item['repository']['full_name']}\n"
        )
    
    return "\n".join(formatted_results)

@github_agent.tool
async def get_commit_history(ctx: RunContext[Deps], github_url: str, days: int = 7) -> str:
    """Get recent commit history for the repository using GitHub API.
    
    Args:
        ctx: The context
        github_url: The GitHub repository URL
        days: Number of days of history to retrieve (default: 7)
    
    Returns:
        str: Commit history as a formatted string
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    # Calculate date for filtering
    since = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/commits',
        params={'since': since},
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get commit history: {response.text}"
    
    commits = response.json()
    formatted_commits = []
    
    for commit in commits:
        formatted_commits.append(
            f"SHA: {commit['sha'][:7]}\n"
            f"Author: {commit['commit']['author']['name']}\n"
            f"Date: {commit['commit']['author']['date']}\n"
            f"Message: {commit['commit']['message']}\n"
        )
    
    return "\n".join(formatted_commits)

@github_agent.tool
async def analyze_code_frequency(ctx: RunContext[Deps], github_url: str) -> str:
    """Analyze code frequency statistics using GitHub API.
    
    Args:
        ctx: The context
        github_url: The GitHub repository URL
    
    Returns:
        str: Code frequency statistics as a formatted string
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/stats/code_frequency',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get code frequency stats: {response.text}"
    
    stats = response.json()
    formatted_stats = ["Weekly Code Frequency (last 4 weeks):"]
    
    # Process last 4 weeks of data
    for week_data in stats[-4:]:
        timestamp, additions, deletions = week_data
        week_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        formatted_stats.append(
            f"Week of {week_date}:\n"
            f"  Additions: +{additions}\n"
            f"  Deletions: {deletions}\n"
        )
    
    return "\n".join(formatted_stats)



class GitHubChat:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.deps = Deps(
            client=httpx.AsyncClient(),
            github_token=os.getenv('GITHUB_TOKEN')
        )

    async def chat(self):
        print("GitHub Agent Chat (type 'quit' to exit)")
        print("How can I help you with GitHub repositories?")
        
        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break

                # Run the agent with the user's input
                result = await github_agent.run(
                    user_input,
                    deps=self.deps,
                    message_history=self.messages
                )

                # Store the messages for context
                self.messages.extend(result.new_messages())
                
                # Print the agent's response
                print(result.data)
                print()

        finally:
            await self.deps.client.aclose()
            
            
async def main():
    chat = GitHubChat()
    await chat.chat()

if __name__ == "__main__":
    asyncio.run(main())