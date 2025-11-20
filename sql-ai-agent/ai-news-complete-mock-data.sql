-- Insert mock data into rss_feeds table
INSERT OR IGNORE INTO rss_feeds (name, url, description, site_link, language) VALUES
('AI Daily', 'https://ai-daily.com/feed', 'Daily AI news and updates', 'https://ai-daily.com', 'en'),
('ML Weekly', 'https://mlweekly.com/rss', 'Weekly roundup of machine learning news', 'https://mlweekly.com', 'en'),
('IA Nouvelles', 'https://ia-nouvelles.fr/flux', 'Actualités sur l''intelligence artificielle en français', 'https://ia-nouvelles.fr', 'fr'),
('Data Science Digest', 'https://datasciencedigest.com/feed', 'Comprehensive coverage of data science topics', 'https://datasciencedigest.com', 'en'),
('AI Ethics Blog', 'https://aiethicsblog.org/rss', 'Exploring ethical implications of AI', 'https://aiethicsblog.org', 'en');

-- Insert mock data into categories table
INSERT OR IGNORE INTO categories (name, description) VALUES
('Machine Learning', 'News related to machine learning algorithms and techniques'),
('Natural Language Processing', 'Updates on NLP research and applications'),
('Computer Vision', 'Advancements in image and video processing using AI'),
('Ethics in AI', 'Discussions on ethical considerations in AI development and deployment'),
('Robotics', 'News about AI in robotics and automation'),
('AI in Healthcare', 'Applications of AI in medicine and healthcare'),
('Deep Learning', 'Focused on deep neural networks and related technologies');

-- Insert mock data into rss_items table
INSERT OR IGNORE INTO rss_items (rss_feed_id, title, link, description, content, published_date, author) VALUES
(1, 'New breakthrough in reinforcement learning', 'https://ai-daily.com/articles/reinforcement-learning-breakthrough', 'Researchers achieve significant progress in RL algorithms', 'Full content of the article...', '2023-04-15 09:30:00', 'Jane Doe'),
(2, 'GPT-4 shows impressive results in medical diagnosis', 'https://mlweekly.com/news/gpt4-medical-diagnosis', 'OpenAI''s latest language model demonstrates potential in healthcare', 'Detailed article content...', '2023-04-14 14:45:00', 'John Smith'),
(3, 'L''IA générative révolutionne la création artistique', 'https://ia-nouvelles.fr/articles/ia-generative-art', 'Comment l''IA transforme le processus créatif des artistes', 'Contenu complet de l''article...', '2023-04-13 11:15:00', 'Marie Dupont'),
(4, 'Advancements in Computer Vision for Autonomous Vehicles', 'https://datasciencedigest.com/articles/cv-autonomous-vehicles', 'Recent developments in CV improving self-driving car capabilities', 'Full article content...', '2023-04-16 10:00:00', 'Alex Johnson'),
(5, 'The Ethics of AI in Hiring Processes', 'https://aiethicsblog.org/posts/ai-in-hiring', 'Examining the implications of using AI for job candidate selection', 'Detailed blog post content...', '2023-04-17 13:20:00', 'Samantha Lee');

-- Insert mock data into rss_item_categories junction table
INSERT OR IGNORE INTO rss_item_categories (rss_item_id, category_id) VALUES
(1, 1), (1, 7), -- Reinforcement learning article tagged with Machine Learning and Deep Learning
(2, 2), (2, 6), -- GPT-4 article tagged with NLP and AI in Healthcare
(3, 1), (3, 4), -- Generative AI article tagged with Machine Learning and Ethics in AI
(4, 3), (4, 5), -- Computer Vision article tagged with Computer Vision and Robotics
(5, 4), (5, 6); -- AI Ethics article tagged with Ethics in AI and AI in Healthcare

-- Insert mock data into users table
INSERT OR IGNORE INTO users (username, email, password_hash, created_at, last_login) VALUES
('alice_ai', 'alice@example.com', 'hashed_password_1', '2023-01-01 10:00:00', '2023-04-15 14:30:00'),
('bob_ml', 'bob@example.com', 'hashed_password_2', '2023-02-15 11:30:00', '2023-04-14 09:15:00'),
('charlie_nlp', 'charlie@example.com', 'hashed_password_3', '2023-03-20 09:45:00', '2023-04-13 16:45:00'),
('dana_cv', 'dana@example.com', 'hashed_password_4', '2023-03-25 14:00:00', '2023-04-16 11:30:00'),
('evan_ethics', 'evan@example.com', 'hashed_password_5', '2023-04-01 08:30:00', '2023-04-17 10:45:00');

-- Insert mock data into user_category_preferences table
INSERT OR IGNORE INTO user_category_preferences (user_id, category_id) VALUES
(1, 1), (1, 2), (1, 7), -- Alice is interested in Machine Learning, NLP, and Deep Learning
(2, 1), (2, 3), (2, 5), -- Bob is interested in Machine Learning, Computer Vision, and Robotics
(3, 2), (3, 4), (3, 6), -- Charlie is interested in NLP, Ethics in AI, and AI in Healthcare
(4, 3), (4, 5), (4, 7), -- Dana is interested in Computer Vision, Robotics, and Deep Learning
(5, 4), (5, 6), (5, 1); -- Evan is interested in Ethics in AI, AI in Healthcare, and Machine Learning

-- Insert mock data into user_feed_preferences table
INSERT OR IGNORE INTO user_feed_preferences (user_id, rss_feed_id) VALUES
(1, 1), (1, 2), (1, 4), -- Alice follows AI Daily, ML Weekly, and Data Science Digest
(2, 2), (2, 3), (2, 4), -- Bob follows ML Weekly, IA Nouvelles, and Data Science Digest
(3, 1), (3, 3), (3, 5), -- Charlie follows AI Daily, IA Nouvelles, and AI Ethics Blog
(4, 1), (4, 2), (4, 4), -- Dana follows AI Daily, ML Weekly, and Data Science Digest
(5, 3), (5, 4), (5, 5); -- Evan follows IA Nouvelles, Data Science Digest, and AI Ethics Blog

-- Insert mock data into article_interactions table
INSERT OR IGNORE INTO article_interactions (user_id, rss_item_id, interaction_type, interaction_time) VALUES
(1, 1, 'view', '2023-04-15 10:15:00'),
(1, 1, 'like', '2023-04-15 10:20:00'),
(1, 2, 'view', '2023-04-15 10:30:00'),
(2, 2, 'view', '2023-04-14 15:00:00'),
(2, 2, 'like', '2023-04-14 15:05:00'),
(2, 2, 'share', '2023-04-14 15:10:00'),
(3, 3, 'view', '2023-04-13 17:00:00'),
(3, 3, 'like', '2023-04-13 17:10:00'),
(4, 4, 'view', '2023-04-16 11:45:00'),
(4, 4, 'share', '2023-04-16 11:50:00'),
(5, 5, 'view', '2023-04-17 14:00:00'),
(5, 5, 'like', '2023-04-17 14:15:00');

-- Insert mock data into feed_views table
INSERT OR IGNORE INTO feed_views (user_id, rss_feed_id, viewed_at) VALUES
(1, 1, '2023-04-15 10:00:00'),
(1, 2, '2023-04-15 10:25:00'),
(2, 2, '2023-04-14 14:55:00'),
(3, 3, '2023-04-13 16:50:00'),
(4, 4, '2023-04-16 11:40:00'),
(5, 5, '2023-04-17 13:55:00');

-- Insert mock data into user_sessions table
INSERT OR IGNORE INTO user_sessions (user_id, session_token, created_at, expires_at) VALUES
(1, 'token_alice_1', '2023-04-15 14:30:00', '2023-04-16 14:30:00'),
(2, 'token_bob_1', '2023-04-14 09:15:00', '2023-04-15 09:15:00'),
(3, 'token_charlie_1', '2023-04-13 16:45:00', '2023-04-14 16:45:00'),
(4, 'token_dana_1', '2023-04-16 11:30:00', '2023-04-17 11:30:00'),
(5, 'token_evan_1', '2023-04-17 10:45:00', '2023-04-18 10:45:00');
