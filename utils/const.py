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