from celery_app import app
from classifier import classify_review
from db import save_result
from utils.logger import logger

@app.task
def classify_review_task(review_id, product_category, review_text, rating):
    """Celery task: classify a review and persist the result to Postgres.

    Parameters
    ----------
    review_id : str
        Unique identifier for the review being classified.
    product_category : str
        Which product category this review belongs to.
    review_text : str
        The raw review text to classify.
    rating : float
        The original star rating from the source data.

    Returns
    -------
    dict
        The classification result, with review_id attached.
    """
    logger.log(f"Starting classification for review_id={review_id}")
    result = classify_review(review_text)
    save_result(review_id, product_category, review_text, rating, result)
    result["review_id"] = review_id
    logger.log(f"Finished and saved review_id={review_id}: {result['feedback_category']}")
    return result