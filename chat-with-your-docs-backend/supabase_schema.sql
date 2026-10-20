-- ==============================================================================
-- DocTalk Supabase Schema & pgvector Setup
-- Run this in your Supabase Dashboard: SQL Editor -> New Query -> Run
-- ==============================================================================

-- 1. Enable the pgvector extension for dense vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the Users Table
CREATE TABLE IF NOT EXISTS public.users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT,
    last_login TEXT
);

-- Index on email for fast authentication lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);

-- 3. Create Document Chunks Table (768 dimensions for nomic-embed-text)
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    doc_name TEXT NOT NULL,
    section_idx INT DEFAULT 0,
    chunk_idx INT DEFAULT 0,
    section_title TEXT DEFAULT '',
    content TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for lightning-fast per-user queries and HNSW vector similarity
CREATE INDEX IF NOT EXISTS idx_chunks_user ON public.document_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON public.document_chunks(user_id, doc_name);
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON public.document_chunks USING hnsw (embedding vector_cosine_ops);

-- 4. Vector Search RPC Function (Cosine Distance)
CREATE OR REPLACE FUNCTION match_document_chunks(
    p_user_id TEXT,
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.0,
    match_count INT DEFAULT 30
)
RETURNS TABLE (
    id UUID,
    doc_name TEXT,
    section_idx INT,
    chunk_idx INT,
    section_title TEXT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.doc_name,
        dc.section_idx,
        dc.chunk_idx,
        dc.section_title,
        dc.content,
        (1 - (dc.embedding <=> query_embedding))::FLOAT AS similarity
    FROM public.document_chunks dc
    WHERE dc.user_id = p_user_id
      AND (1 - (dc.embedding <=> query_embedding)) > match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- 5. Storage Bucket for User Files (PDFs, DOCX, MD)
INSERT INTO storage.buckets (id, name, public)
VALUES ('doctalk-documents', 'doctalk-documents', false)
ON CONFLICT (id) DO NOTHING;
