from dotenv import load_dotenv
from swarm import Agent
import sqlite3
import os

load_dotenv()
model = os.getenv('LLM_MODEL', 'qwen2.5-coder:7b')

conn = sqlite3.connect('rss-feed-database.db')
cursor = conn.cursor()

with open("ai-news-complete-tables.sql", "r") as table_schema_file:
    table_schemas = table_schema_file.read()

def run_sql_select_statement(sql_statement):
    """Executes a SQL SELECT statement and returns the results of running the SELECT. Make sure you have a full SQL SELECT query created before calling this function."""
    print(f"Executing SQL statement: {sql_statement}")
    cursor.execute(sql_statement)
    records = cursor.fetchall()

    if not records:
        return "No results found."
    
    # Get column names
    column_names = [description[0] for description in cursor.description]
    
    # Calculate column widths
    col_widths = [len(name) for name in column_names]
    for row in records:
        for i, value in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(value)))
    
    # Format the results
    result_str = ""
    
    # Add header
    header = " | ".join(name.ljust(width) for name, width in zip(column_names, col_widths))
    result_str += header + "\n"
    result_str += "-" * len(header) + "\n"
    
    # Add rows
    for row in records:
        row_str = " | ".join(str(value).ljust(width) for value, width in zip(row, col_widths))
        result_str += row_str + "\n"
    
    return result_str    

def get_sql_router_agent_instructions():
    return """You are an orchestrator of different SQL data experts and it is your job to
    determine which of the agent is best suited to handle the user's request, 
    and transfer the conversation to that agent."""

def get_sql_agent_instructions():
    return f"""You are a SQL expert who takes in a request from a user for information
    they want to retrieve from the DB, creates a SELECT statement to retrieve the
    necessary information, and then invoke the function to run the query and
    get the results back to then report to the user the information they wanted to know.
    
    Here are the table schemas for the DB you can query:
    
    {table_schemas}

    Write all of your SQL SELECT statements to work 100% with these schemas and nothing else.
    You are always willing to create and execute the SQL statements to answer the user's question.
    """


sql_router_agent = Agent(
    name="Router Agent",
    instructions=get_sql_router_agent_instructions(),
    model="qwen2.5:3b"
)
rss_feed_agent = Agent(
    name="RSS Feed Agent",
    instructions=get_sql_agent_instructions() + "\n\nHelp the user with data related to RSS feeds. Be super enthusiastic about how many great RSS feeds there are in every one of your responses.",
    functions=[run_sql_select_statement],
    model=model
)
user_agent = Agent(
    name="User Agent",
    instructions=get_sql_agent_instructions() + "\n\nHelp the user with data related to users.",
    functions=[run_sql_select_statement],
    model=model
)
analytics_agent = Agent(
    name="Analytics Agent",
    instructions=get_sql_agent_instructions() + "\n\nHelp the user gain insights from the data with analytics. Be super accurate in reporting numbers and citing sources.",
    functions=[run_sql_select_statement],
    model=model
)


def transfer_back_to_router_agent():
    """Call this function if a user is asking about data that is not handled by the current agent."""
    return sql_router_agent

def transfer_to_rss_feeds_agent():
    return rss_feed_agent

def transfer_to_user_agent():
    return user_agent

def transfer_to_analytics_agent():
    return analytics_agent


sql_router_agent.functions = [transfer_to_rss_feeds_agent, transfer_to_user_agent, transfer_to_analytics_agent]
rss_feed_agent.functions.append(transfer_back_to_router_agent)
user_agent.functions.append(transfer_back_to_router_agent)
analytics_agent.functions.append(transfer_back_to_router_agent)