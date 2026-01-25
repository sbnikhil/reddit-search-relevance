"""
Generate synthetic Reddit posts for demo purposes
This allows the project to run without BigQuery access
"""

import json
import random
import yaml
import pysolr
from pathlib import Path

# Sample technical topics and content
TOPICS = [
    "Django",
    "React",
    "PostgreSQL",
    "Docker",
    "Python",
    "JavaScript",
    "AWS",
    "Machine Learning",
    "REST API",
    "MongoDB",
    "Redis",
    "Kubernetes",
]

QUESTION_TEMPLATES = [
    "How do I {} in {}?",
    "Best practices for {} with {}",
    "{} performance optimization tips",
    "Debugging {} issues in {}",
    "{} vs {} comparison",
    "Step by step {} tutorial",
    "{} error: how to fix?",
    "Production-ready {} setup",
]

CODE_SNIPPETS = [
    "```python\ndef example():\n    return True\n```",
    "```javascript\nconst api = async () => { return data; }\n```",
    "<code>SELECT * FROM users WHERE active=true</code>",
    "docker run -d -p 8080:80 nginx",
]

SOLUTION_PHRASES = [
    "This fixed the issue for me:",
    "Solved! Here's what worked:",
    "Step by step solution:",
    "After debugging, I found:",
    "The key was to:",
]


def generate_post(idx, is_high_quality=True):
    """Generate a synthetic Reddit post"""
    topic1 = random.choice(TOPICS)
    topic2 = random.choice([t for t in TOPICS if t != topic1])

    template = random.choice(QUESTION_TEMPLATES)
    title = template.format(topic1, topic2)

    # Generate body based on quality
    if is_high_quality and random.random() > 0.3:
        # High utility post with code
        body = f"{random.choice(SOLUTION_PHRASES)}\n\n"
        body += f"When working with {topic1}, you need to consider {topic2}. "
        body += f"\n\n{random.choice(CODE_SNIPPETS)}\n\n"
        body += "This approach handles edge cases and is production-tested."

        expertise = random.uniform(0.6, 0.95)
        utility = random.uniform(0.7, 1.0)
    else:
        # Lower quality post
        body = f"I'm having issues with {topic1}. "
        body += f"Has anyone tried this? Any help would be appreciated."

        expertise = random.uniform(0.1, 0.5)
        utility = random.uniform(0.0, 0.4)

    return {
        "id": f"post_{idx}",
        "title": title,
        "body": body,
        "expertise_score": round(expertise, 3),
        "utility_score": round(utility, 3),
        "label": 1 if (expertise > 0.5 and utility > 0.5) else 0,
    }


def generate_dataset(num_posts=500, high_quality_ratio=0.3):
    """Generate a full synthetic dataset"""
    posts = []

    high_quality_count = int(num_posts * high_quality_ratio)

    # Generate high quality posts
    for i in range(high_quality_count):
        posts.append(generate_post(i, is_high_quality=True))

    # Generate lower quality posts
    for i in range(high_quality_count, num_posts):
        posts.append(generate_post(i, is_high_quality=False))

    random.shuffle(posts)
    return posts


def save_to_json(posts, filepath="data/sample_posts.json"):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(posts, f, indent=2)
    print(f"Saved {len(posts)} posts to {filepath}")


def index_to_solr(posts, solr_url):
    try:
        solr = pysolr.Solr(solr_url, timeout=10)

        solr_docs = []
        for post in posts:
            solr_docs.append(
                {
                    "id": post["id"],
                    "title_t": post["title"],
                    "body_t": post["body"],
                    "expertise_score": post["expertise_score"],
                    "utility_score": post["utility_score"],
                }
            )

        solr.add(solr_docs)
        print(f"Indexed {len(solr_docs)} documents to Solr")
        return True
    except Exception as e:
        print(f"Error indexing to Solr: {e}")
        return False


def main():
    print("Generating synthetic Reddit dataset")

    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    posts = generate_dataset(num_posts=500, high_quality_ratio=0.3)
    save_to_json(posts, "data/sample_posts.json")

    solr_url = cfg["search_tuning"]["solr_url"]
    indexed = index_to_solr(posts, solr_url)

    if not indexed:
        print("Warning: Solr indexing failed. Start Solr and run: make ingest-samples")


if __name__ == "__main__":
    main()
