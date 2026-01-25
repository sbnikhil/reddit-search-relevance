import pytest
from data.pipelines.utility_extractor import MarkUtility


def test_utility_keyword_detection():
    keywords = ["fixed", "solved"]
    detector = MarkUtility(keywords)
    element = {"author": "user1", "subreddit": "python", "body": "I fixed the bug!"}
    result = list(detector.process(element))
    assert result[0][1] == 1


def test_no_utility_keywords():
    keywords = ["fixed", "solved"]
    detector = MarkUtility(keywords)
    element = {"author": "user2", "subreddit": "python", "body": "Hello world"}
    result = list(detector.process(element))
    assert len(result) == 0


def test_case_insensitive_matching():
    keywords = ["fixed"]
    detector = MarkUtility(keywords)
    element = {"author": "user3", "subreddit": "python", "body": "FIXED the issue"}
    result = list(detector.process(element))
    assert len(result) == 1


def test_multiple_keywords():
    keywords = ["fixed", "solved", "worked"]
    detector = MarkUtility(keywords)
    element = {"author": "user4", "subreddit": "python", "body": "fixed and it worked!"}
    result = list(detector.process(element))
    assert len(result) == 1
