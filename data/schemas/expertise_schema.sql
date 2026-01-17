WITH user_activity AS (
    SELECT 
        author,          
        subreddit,
        COUNT(id) AS post_count,
        SUM(score) AS total_upvotes
    FROM `@source_table`
    WHERE created_utc < @training_cutoff_timestamp 
    AND author IS NOT NULL
    GROUP BY 1, 2
),

global_stats AS (
    SELECT 
        SUM(post_count) AS total_posts,
        SUM(total_upvotes) AS total_global_upvotes
    FROM user_activity
),

final_features AS (
    SELECT 
        u.author,
        u.subreddit,
        u.total_upvotes,
        SAFE_DIVIDE(u.total_upvotes, (SELECT total_global_upvotes FROM global_stats)) AS topical_focus_score,
        LOG(u.post_count + 1) AS activity_density
    FROM user_activity u
    WHERE u.total_upvotes >= @min_upvote_threshold 
)

SELECT * FROM final_features
ORDER BY topical_focus_score DESC;