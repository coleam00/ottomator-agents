# Episodic Memory Migration Instructions for Medical RAG Agent

## Overview

This guide will help you run the database migrations to add enhanced episodic memory capabilities to your Medical RAG Agent system on Supabase. The migration adds:

- **Core episodic memory tables** for tracking conversation episodes
- **Medical entity extraction** for symptoms, conditions, treatments
- **Patient profiles** for aggregated patient information
- **Treatment outcomes tracking** for effectiveness monitoring
- **Symptom timeline** for progression tracking
- **Memory importance scoring** for intelligent memory management

## Prerequisites

Before running the migrations, ensure you have:

1. ✅ Supabase project set up (already configured in your .env)
2. ✅ Access to Supabase Dashboard SQL Editor
3. ✅ Service role key configured (already in your .env)

## Migration Files

The system includes several migration-related files:

1. **`sql/migrations/001_add_episodic_memory_tables.sql`** - Base episodic memory tables
2. **`sql/migrations/002_enhance_episodic_memory.sql`** - Medical entity tracking enhancements
3. **`sql/combined_episodic_memory_migration.sql`** - Combined migration (recommended)

## Step-by-Step Migration Instructions

### Option 1: Using Supabase Dashboard (Recommended)

1. **Open Supabase SQL Editor**
   - Go to: https://supabase.com/dashboard/project/bpopugzfbokjzgawshov/sql
   - You should see the SQL Editor interface

2. **Copy the Migration SQL**
   - Open the file: `sql/combined_episodic_memory_migration.sql`
   - Copy the entire content (Ctrl+A, Ctrl+C or Cmd+A, Cmd+C)

3. **Execute the Migration**
   - Paste the SQL into the Supabase SQL Editor
   - Click the "Run" button (or press Ctrl+Enter / Cmd+Enter)
   - Wait for execution to complete (should take 10-30 seconds)

4. **Verify the Migration**
   ```bash
   python verify_episodic_migration.py
   ```
   - All tables should show "✅ Exists"
   - You should see "All migrations successfully verified!"

### Option 2: Using psql Command Line

1. **Get Database Password**
   - Go to: https://supabase.com/dashboard/project/bpopugzfbokjzgawshov/settings/database
   - Copy your database password

2. **Update .env File**
   ```env
   DATABASE_URL=postgresql://postgres:[YOUR_PASSWORD]@db.bpopugzfbokjzgawshov.supabase.co:5432/postgres
   ```
   Replace `[YOUR_PASSWORD]` with the actual password

3. **Run Migration**
   ```bash
   psql -d "$DATABASE_URL" -f sql/combined_episodic_memory_migration.sql
   ```

4. **Verify**
   ```bash
   python verify_episodic_migration.py
   ```

## What Gets Created

### Migration 001: Base Episodic Memory
- **episodes** - Core episode storage with embeddings
- **episode_references** - Links episodes to documents/chunks
- **episode_relationships** - Relationships between episodes
- **memory_summaries** - Aggregated insights from episodes

### Migration 002: Medical Entity Tracking
- **medical_entities** - Extracted medical entities (symptoms, conditions, etc.)
- **symptom_timeline** - Symptom progression tracking
- **treatment_outcomes** - Treatment effectiveness monitoring
- **episode_medical_facts** - Medical fact triples
- **patient_profiles** - Aggregated patient information
- **memory_importance_scores** - Episode importance scoring

### Database Views
- **episode_summary** - Comprehensive episode information
- **patient_medical_history** - Patient medical history overview

### Functions & Triggers
- Episode importance calculation
- Patient profile updates
- Automatic timestamp updates
- Related episode retrieval

## Verification Script

The `verify_episodic_migration.py` script checks:
- All tables exist
- All views are created
- Row counts for each table
- Overall migration status

Run it with:
```bash
python verify_episodic_migration.py
```

## Migration Scripts Available

1. **`run_supabase_migration.py`** - Prepares migrations for manual execution
2. **`verify_episodic_migration.py`** - Verifies migration success
3. **`sql/run_migration.py`** - Alternative migration runner (requires psql)

## Troubleshooting

### Issue: Tables not showing as created
**Solution**: The Supabase API cache may need refreshing. Wait a few seconds and run verification again.

### Issue: Permission errors
**Solution**: Ensure you're using the service role key, not the anon key.

### Issue: Migration partially fails
**Solution**: 
1. Check which tables were created with verification script
2. Drop partially created tables if needed
3. Re-run the complete migration

### Issue: "relation does not exist" errors
**Solution**: Run Migration 001 before Migration 002, or use the combined migration file.

## Post-Migration Steps

After successful migration:

1. **Test the System**
   ```bash
   python test_supabase_connection.py
   ```

2. **Start the API Server**
   ```bash
   python -m agent.api
   ```

3. **Test with CLI**
   ```bash
   python cli.py
   ```

4. **Monitor Episode Creation**
   - Episodes will be automatically created during conversations
   - Medical entities will be extracted from user messages
   - Treatment outcomes and symptoms will be tracked

## Integration with Existing System

The episodic memory system integrates with:
- **agent/episodic_memory.py** - Episode management and retrieval
- **agent/medical_entities.py** - Medical entity extraction
- **agent/fact_extractor.py** - Fact triple extraction

These components will automatically use the new tables once the migration is complete.

## Important Notes

1. **Backup**: Supabase automatically backs up your database, but consider exporting important data before major migrations.

2. **Idempotency**: The migrations use `CREATE TABLE IF NOT EXISTS` so they can be run multiple times safely.

3. **Performance**: The migration creates all necessary indexes for optimal query performance.

4. **Extensions**: The migration enables required PostgreSQL extensions (uuid-ossp, pg_trgm, vector).

## Support

If you encounter issues:
1. Check the verification script output for specific missing tables
2. Review the Supabase dashboard logs for any SQL errors
3. Ensure all environment variables are properly configured
4. The combined migration file is the most reliable option

## Success Indicators

You'll know the migration was successful when:
- ✅ All tables show as "Exists" in verification script
- ✅ No errors in Supabase SQL Editor
- ✅ The API server starts without database errors
- ✅ Episodes are created during conversations

---

**Ready to proceed?** Start with Option 1 (Supabase Dashboard) for the easiest experience!