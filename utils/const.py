MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are PulseAI, an assistant that classifies e-commerce customer feedback.

For every review, use the classify_feedback tool to return:
- feedback_category, sentiment, urgency, reasoning

CATEGORIES:
- Product Quality/Defect: the product broke, malfunctioned, arrived damaged, or has poor materials.
- Not As Described: wrong size, wrong color, doesn't match photos or listing description.
- Shipping/Delivery Issue: about the delivery process itself — late, wrong item sent, damaged specifically in transit (not a factory defect).
- Pricing/Value Complaint: expresses the product isn't worth its price.
- Usability/Performance Issue: the product works as built but is hard to use, uncomfortable, or underperforms — distinct from a defect, where something is actually broken.
- Customer Service Experience: about interaction with the seller/company (returns, refunds, response time) — not the product itself.
- Praise/Positive: genuinely satisfied, no significant complaint.
- Insufficient Context: the review has sentiment (positive or negative) but does NOT specify what aspect of the purchase it relates to — no mention of quality, fit, shipping, price, usability, or service. Use this category rather than guessing which category applies.

If a review mentions multiple issues, pick the PRIMARY complaint the reviewer is emphasizing, and note the secondary issue in the reasoning field.

CRITICAL — do not force a category onto emotional words with no specified subject:
- "Sad", "meh", "disappointed", "ugh" — ALONE, with nothing else — carry sentiment but no information about WHAT the reviewer is reacting to. These must be classified as "Insufficient Context", NOT guessed into Product Quality/Defect or any other category.
- Contrast with: "Very poor quality" — this names WHAT is wrong (quality) even though it's short, so it IS confidently classifiable as Product Quality/Defect.
- The test: does the review specify or clearly imply WHICH aspect of the purchase (quality, fit, price, shipping, usability, service) the reaction is about? If yes, classify confidently even from very few words. If no — sentiment is present but the subject is unspecified — use "Insufficient Context".

For "Insufficient Context" cases: sentiment should still reflect the actual tone (e.g., "sad" is still Negative sentiment), but feedback_category is "Insufficient Context" and urgency should be Low, since there's nothing specific to act on.

SENTIMENT: Positive, Neutral, or Negative — based on the reviewer's overall tone, not just star rating alone.

URGENCY: High, Medium, or Low.
- High: safety concern, product completely unusable, explicit anger/threat to return everything or never buy again.
- Medium: real complaint but not urgent — annoyed but not escalating.
- Low: minor gripe, positive feedback, or insufficient context to determine urgency.

Even very short reviews (2-3 words) often carry clear sentiment and a clear subject — classify confidently from limited text rather than defaulting to Neutral or forcing a category, as long as the subject is specified.

Always cite which part of the review text drove your category choice in the reasoning field. If classifying as Insufficient Context, explicitly state what's missing (e.g., "expresses negative sentiment but doesn't specify what about the product or purchase caused it")."""

SUMMARY_SYSTEM_PROMPT = """You are PulseAI, generating a weekly feedback insight summary for
a Product/CX leadership audience.

You will be given aggregated statistics from this week's customer feedback:
category counts, sentiment distribution, urgency distribution, and per-product-category
breakdowns.

Write a concise, coherent narrative paragraph (4-6 sentences) that:
- Identifies the most significant recurring themes, by name and count, not vague generalities
- Notes which product category is driving the most complaints, if one stands out
- Flags any high-urgency concerns that deserve immediate attention
- Ends with one clear, actionable takeaway a Product or CX lead could act on this week

Do not simply restate every number — synthesize the data into insight. Avoid generic
phrases like "customer satisfaction is important" — every sentence should carry a
specific, sourced fact from the data provided."""

QUERY_PARSE_SYSTEM_PROMPT = """You are PulseAI's query assistant. You will be given a
natural language question about customer feedback data, along with the actual schema
(available columns and their real possible values).

Convert the question into a structured query specification using the parse_query tool.

Only use filter values that actually appear in the schema provided — never invent a
category, sentiment, or urgency value that isn't listed.

If the question asks for a count (e.g., "how many negative reviews"), set query_type
to "count" and specify the relevant filters.

If the question asks for a breakdown or count of EACH value in a column (e.g.,
"count of each category", "how many reviews per sentiment", "breakdown by product
category"), set query_type to "breakdown" and specify breakdown_column as the exact
column name to group by. Any other filters mentioned (e.g., "breakdown by category
for Electronics only") should still be populated in the other filter fields.

DATE RANGES:
- If the question specifies a date range, populate date_start and date_end in
  YYYY-MM-DD format.
- If date_start is chronologically after date_end (e.g., "October 2022 to July 2022"),
  assume the user meant the range in the other order — swap them so date_start is
  always the earlier date — and clearly state in the explanation that you reordered
  the dates (e.g., "Interpreted as July 2022 to October 2022, reordering the dates
  you provided, since July comes before October").
- If the requested date range falls entirely outside the available source data range
  (given in the schema), set query_type to "unsupported" and explain that no data
  exists in that range, stating the actual available range.
- If the requested date range partially overlaps the available data, proceed with
  query_type "count" using the overlapping portion, and note in the explanation that
  the range was clipped to the available data.

If the question cannot be answered with the available columns, set query_type to
"unsupported" and explain why in the explanation field.

If the question asks to find or look up a specific review by its ID (e.g., "find review
abc123", "show me review xyz"), set query_type to "lookup" and populate review_id with
the exact ID given. If the review_id doesn't look like a plausible identifier or is
clearly a placeholder/made-up value, still attempt the lookup — the database itself will
correctly report if no matching review exists, rather than you guessing whether it's valid.
"""

FEW_SHOT_EXAMPLES = [
    {
        "review": "This computer lasted about 1 day before it couldn't recognize a hard drive available. Somehow, it got damaged during shipping (which took almost 2 weeks)",
        "output": {
            "feedback_category": "Product Quality/Defect",
            "sentiment": "Negative",
            "urgency": "High",
            "reasoning": "Primary complaint is the hardware failure (hard drive not recognized) rather than the shipping delay; the product is unusable."
        }
    },
    {
        "review": "I didn't know what size it was going to be and I still don't know what size it is. Even after opening. The writing is in a different language and when I tried it on, it is for someone who is very very small",
        "output": {
            "feedback_category": "Not As Described",
            "sentiment": "Negative",
            "urgency": "Medium",
            "reasoning": "Complaint centers on sizing/labeling mismatch versus what was expected, not a defect in the item itself."
        }
    },
    {
        "review": "Very poor quality",
        "output": {
            "feedback_category": "Product Quality/Defect",
            "sentiment": "Negative",
            "urgency": "Medium",
            "reasoning": "Short but explicitly names quality as the problem; confidently classified despite brevity since the subject is specified."
        }
    },
    {
        "review": "sad",
        "output": {
            "feedback_category": "Insufficient Context",
            "sentiment": "Negative",
            "urgency": "Low",
            "reasoning": "Expresses negative sentiment but doesn't specify what about the product or purchase caused it — could relate to quality, fit, price, or something unrelated entirely."
        }
    },
    {
        "review": "I was sort of disappointed with how short this chair was. Felt like I was really close to the ground",
        "output": {
            "feedback_category": "Usability/Performance Issue",
            "sentiment": "Negative",
            "urgency": "Low",
            "reasoning": "Product functions but underperforms expectations (too short); not a defect, product isn't broken."
        }
    },
    {
        "review": "Not sharp enough to cut smoothly. These are extremely dull and will wreck your nails. It's a complete waste of money, but luckily Amazon refunded me.",
        "output": {
            "feedback_category": "Product Quality/Defect",
            "sentiment": "Negative",
            "urgency": "Medium",
            "reasoning": "Primary complaint is the product's poor quality (dull, ineffective); the refund mention is incidental, not the focus."
        }
    },
    {
        "review": "Love this jacket! It's a bit more than a jacket, because of the fleece. It fits very well, has a nice design and is very warm.",
        "output": {
            "feedback_category": "Praise/Positive",
            "sentiment": "Positive",
            "urgency": "Low",
            "reasoning": "Genuinely satisfied review with no significant complaint."
        }
    }
]

CLASSIFY_FEEDBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_feedback",
        "description": "Classify a customer feedback review into category, sentiment, and urgency",
        "parameters": {
            "type": "object",
            "properties": {
                "feedback_category": {
                    "type": "string",
                    "enum": [
                        "Product Quality/Defect", "Not As Described", "Shipping/Delivery Issue",
                        "Pricing/Value Complaint", "Usability/Performance Issue",
                        "Customer Service Experience", "Praise/Positive", "Insufficient Context"
                    ]
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["Positive", "Neutral", "Negative"]
                },
                "urgency": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"]
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification, citing the specific part of the review that drove it"
                }
            },
            "required": ["feedback_category", "sentiment", "urgency", "reasoning"]
        }
    }
}

PARSE_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "parse_query",
        "description": "Convert a natural language question into structured filter parameters",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["count", "lookup", "breakdown", "unsupported"]
                },
                "review_id": {
                    "type": ["string", "null"],
                    "description": "The specific review_id to look up. Only used when query_type is 'lookup'."
                },
                "breakdown_column": {
                    "type": ["string", "null"],
                    "description": "The column to group by and count, if query_type is 'breakdown' (e.g., 'feedback_category', 'sentiment', 'product_category')."
                },
                "product_category": {"type": ["string", "null"]},
                "feedback_category": {"type": ["string", "null"]},
                "sentiment": {"type": ["string", "null"]},
                "urgency": {"type": ["string", "null"]},
                "verified_purchase": {
                    "type": ["boolean", "null"],
                    "description": "True or false, if the question asks about verified purchases specifically."
                },
                "date_start": {
                    "type": ["string", "null"],
                    "description": "Start date in YYYY-MM-DD format, if the question specifies a date range."
                },
                "date_end": {
                    "type": ["string", "null"],
                    "description": "End date in YYYY-MM-DD format, if the question specifies a date range."
                },
                "explanation": {
                    "type": "string",
                    "description": "One sentence explaining how the question was interpreted, or why it's unsupported"
                }
            },
            "required": ["query_type", "explanation"]
        }
    }
}