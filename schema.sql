-- Database Schema for Runify API

-- Athletes Table
CREATE TABLE athletes (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    firstname VARCHAR(255) NOT NULL,
    lastname VARCHAR(255) NOT NULL,
    profile_image_url TEXT,
    city VARCHAR(255),
    state VARCHAR(255),
    country VARCHAR(255),
    sex VARCHAR(20),
    bio TEXT,
    location VARCHAR(255),
    default_sport VARCHAR(20),
    weight NUMERIC,
    height INTEGER,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Athlete Followers (Many-to-Many)
CREATE TABLE athlete_followers (
    follower_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    followed_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    PRIMARY KEY (follower_id, followed_id)
);

-- Gear Table
CREATE TABLE gear (
    id VARCHAR(20) PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    brand_name VARCHAR(255),
    model_name VARCHAR(255),
    primary BOOLEAN DEFAULT FALSE,
    type VARCHAR(10) NOT NULL,
    mileage NUMERIC DEFAULT 0
);

-- Activities Table
CREATE TABLE activities (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(20) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    elapsed_time INTEGER NOT NULL,
    distance NUMERIC,
    elevation_gained NUMERIC,
    description TEXT,
    visibility VARCHAR(20) DEFAULT 'everyone',
    whitelist JSONB DEFAULT '[]'::JSONB,  -- Array of athlete IDs for whitelist visibility
    hidden_stats JSONB DEFAULT '[]'::JSONB,  -- Array of strings
    tags JSONB DEFAULT '[]'::JSONB,  -- Array of strings
    gear_id VARCHAR(20) REFERENCES gear(id),
    perceived_exertion INTEGER CHECK (perceived_exertion BETWEEN 1 AND 10),
    manual BOOLEAN DEFAULT FALSE,
    city VARCHAR(255),
    state VARCHAR(255),
    country VARCHAR(255),
    invisible BOOLEAN DEFAULT FALSE,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    average_speed NUMERIC,
    max_speed NUMERIC,
    average_watts NUMERIC,
    maximum_watts NUMERIC,
    calories_burnt NUMERIC,
    average_cadence NUMERIC,
    max_cadence NUMERIC,
    average_heartrate NUMERIC,
    max_heartrate NUMERIC,
    min_elevation NUMERIC,
    max_elevation NUMERIC,
    grade_adjusted_pace NUMERIC,
    relative_effort NUMERIC,
    polyline TEXT,
    summary_polyline TEXT
);

-- Laps Table
CREATE TABLE laps (
    id SERIAL PRIMARY KEY,
    activity_id INTEGER REFERENCES activities(id) ON DELETE CASCADE,
    lap_index INTEGER NOT NULL,
    start_index INTEGER,
    end_index INTEGER,
    distance NUMERIC,
    elapsed_time INTEGER,
    moving_time INTEGER,
    average_speed NUMERIC,
    max_speed NUMERIC,
    elevation_gain NUMERIC,
    start_time TIMESTAMP
);

-- Activity Zones Table
CREATE TABLE activity_zones (
    id SERIAL PRIMARY KEY,
    activity_id INTEGER REFERENCES activities(id) ON DELETE CASCADE,
    type VARCHAR(20) NOT NULL,  -- heartrate or power
    sensor_based BOOLEAN DEFAULT TRUE,
    distribution_buckets JSONB  -- Array of {min: number, max: number, time: integer}
);

-- Activity Streams Table (using JSONB for data arrays)
CREATE TABLE activity_streams (
    id SERIAL PRIMARY KEY,
    activity_id INTEGER REFERENCES activities(id) ON DELETE CASCADE,
    stream_type VARCHAR(20) NOT NULL,  -- time, distance, etc.
    data JSONB NOT NULL,  -- Array of numbers or [number, number] for position
    series_type VARCHAR(20),
    original_size INTEGER,
    resolution VARCHAR(10) DEFAULT 'medium'
);

-- Athlete Zones Table
CREATE TABLE athlete_zones (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    type VARCHAR(20) NOT NULL,  -- heartrate or power
    zones JSONB NOT NULL  -- Array of {min: integer, max: integer, name: string}
);

-- Routes Table
CREATE TABLE routes (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    distance NUMERIC,
    elevation_gain NUMERIC,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    polyline TEXT,
    summary_polyline TEXT
);

-- Route Streams Table (similar to activity streams)
CREATE TABLE route_streams (
    id SERIAL PRIMARY KEY,
    route_id INTEGER REFERENCES routes(id) ON DELETE CASCADE,
    stream_type VARCHAR(20) NOT NULL,
    data JSONB NOT NULL,
    series_type VARCHAR(20),
    original_size INTEGER,
    resolution VARCHAR(10) DEFAULT 'medium'
);

-- Clubs Table
CREATE TABLE clubs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    profile_medium TEXT,
    cover_photo TEXT,
    activities JSONB DEFAULT '[]'::JSONB,  -- Array of activity types
    city VARCHAR(255),
    state VARCHAR(255),
    country VARCHAR(255),
    private BOOLEAN DEFAULT FALSE,
    url VARCHAR(255) UNIQUE,
    tags JSONB DEFAULT '[]'::JSONB,
    member_count INTEGER DEFAULT 0,
    verified BOOLEAN DEFAULT FALSE,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Club Members (Many-to-Many with roles)
CREATE TABLE club_members (
    club_id INTEGER REFERENCES clubs(id) ON DELETE CASCADE,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    admin BOOLEAN DEFAULT FALSE,
    owner BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (club_id, athlete_id)
);

-- Club Posts Table
CREATE TABLE club_posts (
    id SERIAL PRIMARY KEY,
    club_id INTEGER REFERENCES clubs(id) ON DELETE CASCADE,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    link TEXT,
    embed_image_url TEXT,
    tags JSONB DEFAULT '[]'::JSONB,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reports Table
CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    reporter_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    target_type VARCHAR(20) NOT NULL,  -- activity, athlete, route, club
    target_id INTEGER NOT NULL,
    reason VARCHAR(255) NOT NULL,
    description TEXT,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance?
CREATE INDEX idx_activities_athlete_id ON activities(athlete_id);
CREATE INDEX idx_activities_start_time ON activities(start_time);
CREATE INDEX idx_routes_athlete_id ON routes(athlete_id);
CREATE INDEX idx_club_members_club_id ON club_members(club_id);
CREATE INDEX idx_club_members_athlete_id ON club_members(athlete_id);