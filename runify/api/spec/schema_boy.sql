-- Enums
CREATE TYPE activity_type AS ENUM ('run', 'ride', 'swim', 'walk', 'hike', 'workout', 'virtual_run', 'virtual_ride');
CREATE TYPE visibility_level AS ENUM ('followers', 'friends', 'me', 'everyone', 'whitelist');
CREATE TYPE gear_type AS ENUM ('shoes', 'bike');
CREATE TYPE stream_type AS ENUM ('time', 'distance', 'position', 'altitude', 'heartrate', 'cadence', 'power', 'temperature');
CREATE TYPE zone_type AS ENUM ('heartrate', 'power');
CREATE TYPE report_type AS ENUM ('activity', 'athlete', 'route', 'club');
CREATE TYPE report_status AS ENUM ('pending', 'resolved', 'dismissed');
CREATE TYPE unit_preference AS ENUM ('imperial', 'metric');

-- Athletes table
CREATE TABLE athletes (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,  -- Hashed password
    email TEXT UNIQUE NOT NULL,
    firstname TEXT NOT NULL,
    lastname TEXT NOT NULL,
    profile_image_url TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    sex TEXT CHECK (sex IN ('M', 'F', 'cardboard_box')),
    bio TEXT,
    location TEXT,
    default_sport activity_type,
    height INTEGER,  -- cm
    weight NUMERIC,  -- kg
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB  -- e.g., {"units": "metric"}
);

-- Gear table
CREATE TABLE gear (
    id TEXT PRIMARY KEY,  -- e.g., 'g123456'
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    brand_name TEXT,
    model_name TEXT,
    primary_gear BOOLEAN DEFAULT FALSE,
    type gear_type NOT NULL,
    mileage NUMERIC DEFAULT 0  -- meters
);

-- Activities table
CREATE TABLE activities (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type activity_type NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    elapsed_time INTEGER NOT NULL,  -- seconds
    distance NUMERIC,  -- meters
    elevation_gained NUMERIC,  -- meters
    description TEXT,
    visibility visibility_level DEFAULT 'everyone',
    whitelist_athlete_ids INTEGER[],  -- For whitelist visibility
    hidden_stats TEXT[],
    tags TEXT[],
    gear_id TEXT REFERENCES gear(id),
    perceived_exertion INTEGER CHECK (perceived_exertion BETWEEN 1 AND 10),
    manual BOOLEAN DEFAULT FALSE,
    city TEXT,
    state TEXT,
    country TEXT,
    invisible BOOLEAN DEFAULT FALSE,
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    grade_adjusted_pace NUMERIC,  -- seconds
    relative_effort NUMERIC
);

-- Laps table
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
    start_time TIMESTAMP WITH TIME ZONE
);

-- Maps table (for activities and routes)
CREATE TABLE maps (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER NOT NULL,  -- activity_id or route_id
    entity_type TEXT NOT NULL CHECK (entity_type IN ('activity', 'route')),
    polyline TEXT,
    summary_polyline TEXT
);

-- Streams table (JSONB for data arrays)
CREATE TABLE streams (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER NOT NULL,  -- activity_id or route_id
    entity_type TEXT NOT NULL CHECK (entity_type IN ('activity', 'route')),
    stream_type stream_type NOT NULL,
    data JSONB NOT NULL,  -- e.g., array of numbers or [lat, lng]
    series_type TEXT,
    original_size INTEGER,
    resolution TEXT CHECK (resolution IN ('low', 'medium', 'high'))
);

-- Athlete zones
CREATE TABLE athlete_zones (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    type zone_type NOT NULL,
    zones JSONB NOT NULL  -- array of {min, max, name}
);

-- Activity zones
CREATE TABLE activity_zones (
    id SERIAL PRIMARY KEY,
    activity_id INTEGER REFERENCES activities(id) ON DELETE CASCADE,
    type zone_type NOT NULL,
    distribution_buckets JSONB NOT NULL,  -- array of {min, max, time}
    sensor_based BOOLEAN
);

-- Routes table
CREATE TABLE routes (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    distance NUMERIC,
    elevation_gain NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Clubs table
CREATE TABLE clubs (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    profile_medium TEXT,
    cover_photo TEXT,
    activities activity_type[],
    city TEXT,
    state TEXT,
    country TEXT,
    private BOOLEAN DEFAULT FALSE,
    url TEXT UNIQUE,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Club members (junction table)
CREATE TABLE club_members (
    club_id INTEGER REFERENCES clubs(id) ON DELETE CASCADE,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    admin BOOLEAN DEFAULT FALSE,
    owner BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (club_id, athlete_id)
);

-- Club posts
CREATE TABLE club_posts (
    id SERIAL PRIMARY KEY,
    club_id INTEGER REFERENCES clubs(id) ON DELETE CASCADE,
    athlete_id INTEGER REFERENCES athletes(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    link TEXT,
    embed_image_url TEXT,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Follows (athlete follows)
CREATE TABLE follows (
    follower_id INTEGER REFERENCES athletes(id),
    followed_id INTEGER REFERENCES athletes(id),
    PRIMARY KEY (follower_id, followed_id)
);

-- Reports
CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    reporter_id INTEGER REFERENCES athletes(id),
    reported_type report_type NOT NULL,
    reported_id INTEGER NOT NULL,  -- ID of activity/athlete/route/club
    reason TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status report_status DEFAULT 'pending',
    resolved_by INTEGER REFERENCES athletes(id),  -- Admin who resolved
    resolution_notes TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Athlete metrics (time-series)
CREATE TABLE athlete_metrics (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    metric_type TEXT NOT NULL,  -- e.g., 'chronic_training_load'
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value NUMERIC NOT NULL
);

-- Personal records
CREATE TABLE personal_records (
    id SERIAL PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id) ON DELETE CASCADE,
    distance INTEGER NOT NULL,  -- meters
    time NUMERIC NOT NULL,  -- seconds
    name TEXT  -- e.g., '5k'
);

-- Sessions (for auth)
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    athlete_id INTEGER REFERENCES athletes(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX idx_activities_athlete_id ON activities(athlete_id);
CREATE INDEX idx_activities_start_time ON activities(start_time);
CREATE INDEX idx_routes_athlete_id ON routes(athlete_id);
CREATE INDEX idx_club_members_club_id ON club_members(club_id);
CREATE INDEX idx_club_members_athlete_id ON club_members(athlete_id);
CREATE INDEX idx_follows_follower_id ON follows(follower_id);
CREATE INDEX idx_follows_followed_id ON follows(followed_id);
CREATE INDEX idx_reports_status ON reports(status);
CREATE INDEX idx_athlete_metrics_athlete_id_timestamp ON athlete_metrics(athlete_id, timestamp);