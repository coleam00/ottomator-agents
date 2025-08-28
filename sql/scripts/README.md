# SQL Scripts Directory

This directory contains Python scripts for SQL deployment and schema management.

## Scripts

### Deployment Scripts

- **`deploy_schema_via_api.py`** - Deploy database schema using Supabase API
- **`deploy_to_supabase.py`** - Main Supabase deployment script with comprehensive schema setup
- **`run_migration.py`** - SQL migration runner for executing migration files

## Usage

### Deploy Schema to Supabase

```bash
# Deploy using API
python sql/scripts/deploy_schema_via_api.py

# Deploy with comprehensive setup
python sql/scripts/deploy_to_supabase.py
```

### Run Migrations

```bash
# Run specific migration
python sql/scripts/run_migration.py
```

## Related SQL Files

These scripts work with SQL files in the parent directories:

- `../schema.sql` - Main database schema
- `../supabase_schema.sql` - Supabase-specific schema with vector support
- `../migrations/` - Individual migration files
- `../verify_setup.sql` - Schema verification queries

## Environment Requirements

Ensure the following environment variables are set:

- `DATABASE_URL` - PostgreSQL connection string
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key for admin operations

## Notes

- All scripts handle vector dimensions standardized to 1536 (Supabase IVFFlat limit)
- Scripts include proper error handling and rollback support
- Path resolution is handled automatically relative to script location