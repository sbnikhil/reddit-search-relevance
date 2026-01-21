import pytest
from data.pipelines.utility_extractor import MarkUtility

def test_utility_keyword_detection():
    keywords = ["fixed", "solved"]
    detector = MarkUtility(keywords)
    element = {'author': 'user1', 'subreddit': 'python', 'body': 'I fixed the bug!'}
    result = list(detector.process(element))
    assert result[0][1] == 1 
    element = {'author': 'user2', 'subreddit': 'python', 'body': 'Hello world'}
    result = list(detector.process(element))
    assert len(result) == 0 