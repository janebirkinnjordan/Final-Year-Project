-- ============================================================
-- MCP-Powered CRS – Supabase database migration
-- Run this once in: Supabase Dashboard → SQL Editor → New query
-- ============================================================

create table if not exists conversations (
    id         uuid        primary key default gen_random_uuid(),
    public_id  text        not null unique,
    title      text        not null default 'New conversation',
    created_at timestamptz not null default now()
);

create table if not exists messages (
    id         uuid        primary key default gen_random_uuid(),
    session_id text        not null references conversations (public_id) on delete cascade,
    role       varchar(32) not null,
    content    text        not null,
    created_at timestamptz not null default now()
);