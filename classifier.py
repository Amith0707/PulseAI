import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal
from utils.logger import logger
from utils.const import MODEL, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, CLASSIFY_FEEDBACK_TOOL, SUMMARY_SYSTEM_PROMPT, QUERY_PARSE_SYSTEM_PROMPT, PARSE_QUERY_TOOL,FOLLOWUP_SYSTEM_PROMPT

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

def parse_natural_language_query(question: str, schema: dict) -> dict:
    """Convert a natural language question into structured filter parameters.

    Parameters
    ----------
    question : str
        The user's plain-English question.
    schema : dict
        Schema info from :func:`schema_info.get_schema_info`, used to
        ground the LLM in real, existing filter values.

    Returns
    -------
    dict
        Keys: query_type, product_category, feedback_category, sentiment,
        urgency (any of the filter fields may be None), and explanation.
    """
    schema_text = f"""Available columns and their real values:
{schema['columns']}
Total rows in dataset: {schema['row_count']}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": QUERY_PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{schema_text}\n\nQuestion: {question}"}
            ],
            tools=[PARSE_QUERY_TOOL],
            tool_choice={"type": "function", "function": {"name": "parse_query"}},
            temperature=0.1
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    except Exception as e:
        logger.log(f"Query parsing failed: {e}", level="error")
        return {"query_type": "unsupported", "explanation": "Unable to process this question due to a system error."}

def generate_summary(stats: dict) -> str:
    """Generate a narrative weekly insight summary from aggregated stats.

    Unlike :func:`classify_review`, this call does not use tool-calling
    or schema enforcement, since the desired output is free-form prose,
    not structured data.

    Parameters
    ----------
    stats : dict
        Aggregated statistics, as returned by
        :func:`aggregation.get_aggregated_stats` — expects keys
        ``category_counts``, ``sentiment_counts``, ``urgency_counts``,
        and ``product_category_counts``.

    Returns
    -------
    str
        A coherent narrative paragraph summarizing the week's feedback.

    Examples
    --------
    >>> from aggregation import get_aggregated_stats
    >>> stats = get_aggregated_stats()
    >>> summary = generate_summary(stats)
    >>> len(summary) > 0
    True
    """
    stats_text = f"""Feedback category counts: {stats['category_counts']}
Sentiment distribution: {stats['sentiment_counts']}
Urgency distribution: {stats['urgency_counts']}
Reviews by product category: {stats['product_category_counts']}
Total reviews this week: {stats['total_count']}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": stats_text}
            ],
            temperature=0.3
        )
        summary = response.choices[0].message.content
        logger.log("Weekly summary generated successfully")
        return summary
    except Exception as e:
        logger.log(f"Summary generation failed: {e}", level="error")
        return "Unable to generate summary this week due to a system error. Raw statistics are available in the dashboard charts above."

def suggest_followup_questions(narrative: str, stats: dict) -> list:
    """Generate 3 relevant follow-up questions based on a summary just shown.

    Parameters
    ----------
    narrative : str
        The summary narrative just displayed to the user.
    stats : dict
        The underlying stats that generated it.

    Returns
    -------
    list of str
        3 suggested follow-up questions. Returns a safe generic fallback
        list if generation fails.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summary: {narrative}\n\nStats: {stats}"}
            ],
            temperature=0.4
        )
        text = response.choices[0].message.content
        questions = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
        return questions[:3]
    except Exception as e:
        logger.log(f"Follow-up generation failed: {e}", level="error")
        return [
            "How does this compare to the previous period?",
            "Which product category has the most urgent issues?",
            "What's the sentiment breakdown for negative reviews?"
        ]

if __name__ == "__main__":
    test_result = classify_review("This broke after one use, terrible quality for the price")
    print("-" * 30)
    print(test_result)