-- Create the table for processed n8n workflows and their summaries
CREATE TABLE workflows (
    workflow_id INTEGER PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    workflow_description TEXT NOT NULL,
    workflow_json JSONB NOT NULL,
    n8n_demo TEXT NOT NULL,
    summary_accomplishment TEXT NOT NULL,
    summary_nodes TEXT NOT NULL,
    summary_suggestions TEXT NOT NULL,
    embedding vector(1536),
    content TEXT NOT NULL,
    metadata JSONB
);

DROP FUNCTION IF EXISTS match_summaries(vector, integer, jsonb);

-- Create a function to search for summaries
create function match_summaries (
  query_embedding vector(1536),
  match_count int default null,
  filter jsonb DEFAULT '{}'
) returns table (
  id integer,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    workflow_id as id,
    content,
    metadata,
    1 - (workflows.embedding <=> query_embedding) as similarity
  from workflows
  where metadata @> filter
  order by workflows.embedding <=> query_embedding
  limit match_count;
end;
$$;
