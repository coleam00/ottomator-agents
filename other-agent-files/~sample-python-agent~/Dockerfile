FROM ottomator/base-python:latest

# Build argument for port with default value
ARG PORT=8001
ENV PORT=${PORT}

WORKDIR /app

# Copy the application code
COPY . .

# Expose the port from build argument
EXPOSE ${PORT}

# Command to run the application
# Feel free to change sample_supabase_agent to sample_postgres_agent
CMD ["sh", "-c", "uvicorn sample_supabase_agent:app --host 0.0.0.0 --port ${PORT}"]
