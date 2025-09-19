
# init.sql - PostgreSQL Initialization
-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_file_uploads_created_at 
ON file_uploads(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processing_jobs_status 
ON processing_jobs(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processing_jobs_created_at 
ON processing_jobs(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_data_tables_table_name 
ON data_tables(table_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_insights_table_name 
ON ai_insights(table_name);

-- Create a function to clean up old processing logs
CREATE OR REPLACE FUNCTION cleanup_old_logs()
RETURNS void AS $
BEGIN
    -- Delete processing jobs older than 30 days
    DELETE FROM processing_jobs 
    WHERE created_at < NOW() - INTERVAL '30 days' 
    AND status IN ('completed', 'failed');
    
    -- Delete old AI insights (keep last 100 per table)
    WITH ranked_insights AS (
        SELECT id, ROW_NUMBER() OVER (PARTITION BY table_name ORDER BY generated_at DESC) as rn
        FROM ai_insights
    )
    DELETE FROM ai_insights 
    WHERE id IN (
        SELECT id FROM ranked_insights WHERE rn > 100
    );
END;
$ LANGUAGE plpgsql;

-- Schedule cleanup (requires pg_cron extension in production)
-- SELECT cron.schedule('cleanup-old-logs', '0 2 * * *', 'SELECT cleanup_old_logs();');
