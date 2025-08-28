# Scripts Directory

This directory contains utility scripts for deployment, migration, and maintenance of the Medical RAG system.

## Organization

### Main Scripts

- **`start.py`** - Backup startup script for Render deployment with proper module path setup
- **`verify_deployment.py`** - Verify the deployment environment and check all required dependencies

### Database Migration Scripts

- **`run_migration.py`** - General database migration runner for PostgreSQL
- **`run_supabase_migration.py`** - Supabase-specific migration runner with comprehensive migration support
- **`check_and_deploy_episodic_migration.py`** - Check and deploy episodic memory tables migration
- **`verify_episodic_migration.py`** - Verify episodic memory migration status and integrity

### Integration Deployment

- **`deploy_neo4j_integration.py`** - Deploy Neo4j integration including database migrations and Edge Functions

## Usage Examples

### Running Migrations

```bash
# Run standard migrations
python scripts/run_migration.py

# Run Supabase migrations
python scripts/run_supabase_migration.py

# Check and deploy episodic memory migration
python scripts/check_and_deploy_episodic_migration.py

# Verify episodic migration
python scripts/verify_episodic_migration.py
```

### Deployment Verification

```bash
# Verify deployment environment
python scripts/verify_deployment.py

# Deploy Neo4j integration
python scripts/deploy_neo4j_integration.py
```

### Starting the Application

```bash
# Start the application (backup method)
python scripts/start.py
```

## Environment Requirements

All scripts require proper environment variables to be set. Copy `.env.example` to `.env` and configure:

- **Database**: `DATABASE_URL` or Supabase credentials
- **Neo4j**: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- **API Keys**: LLM and embedding provider keys

## Notes

- All scripts now properly reference the parent directory for accessing project resources
- Migration scripts automatically handle path resolution for SQL files
- Scripts include comprehensive error handling and logging