from celery_app import app
from classifier import classify_review
from utils.logger import logger

@app.task
def classify_review_task(review_id, review_text):
    """Celery task wrapping classify_review for async execution.

    Parameters
    ----------
    review_id : str
        Unique identifier for the review being classified.
    review_text : str
        The raw review text to classify.

    Returns
    -------
    dict
        The classification result, with review_id attached.
    """
    logger.log(f"Starting classification for review_id={review_id}")
    result = classify_review(review_text)
    result["review_id"] = review_id
    logger.log(f"Finished review_id={review_id}: {result['feedback_category']}")
    return result