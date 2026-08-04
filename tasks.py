"""
This file defines the task Celery will run. When .delay() is called (from your notebook, once per review), it doesn't run immediately — it gets queued in Redis. 
Whenever one of your 4 worker threads is free, it picks up a queued task and actually executes the function body: calls classifier.py to do the real LLM classification, 
then calls save_result() to write that classification into the classifications table, referencing the review by review_id. 
Celery+Redis's role is entirely about scheduling and parallelizing when/how many of these run at once — the actual classification and database-writing logic itself is just plain Python, 
running the same way it always would, once a thread picks it up.
"""

from celery_app import app
from classifier import classify_review
from db import save_result
from utils.logger import logger

@app.task
def classify_review_task(review_id, review_text):
    """Celery task: classify a review and persist the result.

    Assumes the review itself has already been ingested into the
    reviews table via save_review — this task only writes to
    classifications, referencing the review by review_id.

    Parameters
    ----------
    review_id : str
        Unique identifier for the review being classified (must
        already exist in the reviews table).
    review_text : str
        The raw review text to classify.

    Returns
    -------
    dict
        The classification result, with review_id attached.
    """
    logger.log(f"Starting classification for review_id={review_id}")
    result = classify_review(review_text)
    save_result(review_id, result)
    result["review_id"] = review_id
    logger.log(f"Finished and saved review_id={review_id}: {result['feedback_category']}")
    return result