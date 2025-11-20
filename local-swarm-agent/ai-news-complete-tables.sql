-- Create the RSS feeds table
CREATE TABLE IF NOT EXISTS rss_feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    url VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    site_link VARCHAR(255),
    language VARCHAR(50)
);

-- Create the categories table
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);

-- Create the RSS news feed items table
CREATE TABLE IF NOT EXISTS rss_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rss_feed_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    link VARCHAR(255) NOT NULL,
    description TEXT,
    content TEXT,
    published_date DATETIME,
    author VARCHAR(255),
    FOREIGN KEY (rss_feed_id) REFERENCES rss_feeds(id)
);

-- Create a junction table for the many-to-many relationship between rss_items and categories
CREATE TABLE IF NOT EXISTS rss_item_categories (
    rss_item_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    PRIMARY KEY (rss_item_id, category_id),
    FOREIGN KEY (rss_item_id) REFERENCES rss_items(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
);

-- Create user preferences table for categories
CREATE TABLE IF NOT EXISTS user_category_preferences (
    user_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    PRIMARY KEY (user_id, category_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- Create user preferences table for RSS feeds
CREATE TABLE IF NOT EXISTS user_feed_preferences (
    user_id INTEGER NOT NULL,
    rss_feed_id INTEGER NOT NULL,
    PRIMARY KEY (user_id, rss_feed_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (rss_feed_id) REFERENCES rss_feeds(id)
);

-- Create table for tracking article interactions
CREATE TABLE IF NOT EXISTS article_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    rss_item_id INTEGER NOT NULL,
    interaction_type VARCHAR(20) NOT NULL,  -- 'view', 'like', 'share', etc.
    interaction_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (rss_item_id) REFERENCES rss_items(id)
);

-- Create table for tracking feed views
CREATE TABLE IF NOT EXISTS feed_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    rss_feed_id INTEGER NOT NULL,
    viewed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (rss_feed_id) REFERENCES rss_feeds(id)
);

-- Create table for user sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
