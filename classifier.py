import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal
from utils.logger import logger
from utils.const import MODEL, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, CLASSIFY_FEEDBACK_TOOL

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class FeedbackClassification(BaseModel):
    """Schema for a validated feedback classification result.

    Enforces that every classified review conforms to a fixed set of
    categories, sentiment values, and urgency levels before it is trusted
    or persisted. Used to validate the raw JSON returned by the LLM's tool
    call in :func:`classify_review`.

    Attributes
    ----------
    feedback_category : str
        The feedback category. One of: "Product Quality/Defect",
        "Not As Described", "Shipping/Delivery Issue",
        "Pricing/Value Complaint", "Usability/Performance Issue",
        "Customer Service Experience", "Praise/Positive",
        "Insufficient Context".
    sentiment : str
        Overall reviewer tone. One of: "Positive", "Neutral", "Negative".
    urgency : str
        Priority level for follow-up. One of: "High", "Medium", "Low".
    reasoning : str
        A one-sentence explanation of the classification, citing the
        specific part of the review that drove the decision.
    """
    feedback_category: Literal[
        "Product Quality/Defect", "Not As Described", "Shipping/Delivery Issue",
        "Pricing/Value Complaint", "Usability/Performance Issue",
        "Customer Service Experience", "Praise/Positive", "Insufficient Context"
    ]
    sentiment: Literal["Positive", "Neutral", "Negative"]
    urgency: Literal["High", "Medium", "Low"]
    reasoning: str


FALLBACK_RESULT = {
    "feedback_category": "Insufficient Context",
    "sentiment": "Neutral",
    "urgency": "Low",
    "reasoning": "Automated classification failed due to a system error after retries. This review has NOT been reviewed for content and should be re-processed."
}
"""dict: Safe default result returned by :func:`classify_review` when all
retry attempts are exhausted, either due to LLM API failures or schema
validation failures. Uses "Insufficient Context" rather than forcing a
guessed category, since a system failure carries no real classification
signal.
"""


def build_few_shot_messages():
    """Construct the few-shot example portion of the conversation.

    Converts the static :data:`utils.const.FEW_SHOT_EXAMPLES` list into a
    sequence of user/assistant message pairs, formatted the way the model
    would naturally produce a tool call, so the examples demonstrate the
    exact input/output pattern expected for real reviews.

    Returns
    -------
    list of dict
        A flat list of alternating user and assistant messages, ready to
        be inserted into the `messages` list of a chat completion call.

    Examples
    --------
    >>> messages = _build_few_shot_messages()
    >>> messages[0]["role"]
    'user'
    """
    messages = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f'Review to classify:\n"""{example["review"]}"""'
        })
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"example_{len(messages)}",
                "type": "function",
                "function": {
                    "name": "classify_feedback",
                    "arguments": json.dumps(example["output"])
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"example_{len(messages) - 1}",
            "content": "Recorded."
        })
    return messages


def classify_review(review_text: str, max_retries: int = 2) -> dict:
    """Classify a single customer feedback review.

    Builds a conversation combining the system prompt, few-shot examples,
    and the target review, then calls the LLM with a schema-enforced tool
    call to produce a structured classification. The result is validated
    against :class:`FeedbackClassification` before being returned.

    On failure — whether an LLM API error (rate limit, timeout, connection
    error) or a schema validation failure — the call is retried up to
    ``max_retries`` times with exponential backoff. If all retries fail,
    a safe default (:data:`FALLBACK_RESULT`) is returned instead of
    raising, so the caller never receives an unhandled exception.

    Parameters
    ----------
    review_text : str
        The raw customer review text to classify.
    max_retries : int, optional
        Maximum number of retry attempts after the first failed call
        (default is 2, giving 3 total attempts).

    Returns
    -------
    dict
        A dictionary with keys ``feedback_category``, ``sentiment``,
        ``urgency``, and ``reasoning``, matching
        :class:`FeedbackClassification`. Returns :data:`FALLBACK_RESULT`
        if all attempts fail.

    Examples
    --------
    >>> result = classify_review("This broke after one use, terrible quality")
    >>> result["feedback_category"]
    'Product Quality/Defect'
    """
    import time

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(build_few_shot_messages())
    messages.append({
        "role": "user",
        "content": f'Review to classify:\n"""{review_text}"""'
    })

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[CLASSIFY_FEEDBACK_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_feedback"}},
                temperature=0.1
            )

            tool_call = response.choices[0].message.tool_calls[0]
            raw_result = json.loads(tool_call.function.arguments)
            validated = FeedbackClassification(**raw_result)

            logger.log(f"Classified review successfully: {validated.feedback_category}")
            return validated.model_dump()

        except ValidationError as e:
            logger.log(f"Validation failed on attempt {attempt + 1}: {e}", level="warning")
            if attempt < max_retries:
                continue
            logger.log("Max retries reached after validation failures, using fallback", level="error")
            return FALLBACK_RESULT

        except Exception as e:
            logger.log(f"API call failed on attempt {attempt + 1}: {e}", level="error")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            logger.log("Max retries reached after API failures, using fallback", level="error")
            return FALLBACK_RESULT


if __name__ == "__main__":
    test_result = classify_review("This broke after one use, terrible quality for the price")
    print("-" * 30)
    print(test_result)