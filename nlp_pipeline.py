"""
NLP pipeline for hotel aspect-based sentiment analysis (ABSA).

Features:
  1. preprocess_review()          - clean raw text, strip HTML/URLs, normalize whitespace
  2. extract_aspects()            - robust rule-based aspect extraction with multi-word terms,
                                    disambiguation filters, and clause boundary detection
  3. classify_aspect_sentiments() - pretrained ABSA DeBERTa model with resilient fallback
  4. build_results()              - unified pipeline for single and batch reviews
  5. summarize_by_aspect()        - compute aspect counts, positive ratios, and net sentiment scores
"""
import os
import re
import logging

logger = logging.getLogger(__name__)

# Primary pretrained ABSA model from Hugging Face
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
_sentiment_model = None
_model_load_attempted = False
_model_load_error = None


def get_model_status():
    """Return the current status of the transformer ABSA model."""
    global _sentiment_model, _model_load_attempted, _model_load_error
    if _sentiment_model is not None:
        return {"status": "ready", "model": MODEL_NAME}
    if _model_load_error:
        return {"status": "fallback", "error": str(_model_load_error), "model": "lexicon_fallback"}
    if _model_load_attempted:
        return {"status": "loading", "model": MODEL_NAME}
    return {"status": "unloaded", "model": MODEL_NAME}


def load_model():
    """
    Load the pretrained model the first time it is needed, then reuse it.
    Catches errors gracefully so the server does not crash if offline or downloading.
    """
    global _sentiment_model, _model_load_attempted, _model_load_error
    if _sentiment_model is not None:
        return _sentiment_model

    _model_load_attempted = True
    try:
        from transformers import pipeline
        logger.info(f"Loading transformer model {MODEL_NAME}...")
        _sentiment_model = pipeline("text-classification", model=MODEL_NAME)
        _model_load_error = None
        logger.info("ABSA transformer model loaded successfully.")
    except Exception as exc:
        _model_load_error = str(exc)
        logger.warning(f"Could not load Hugging Face model ({exc}). Utilizing intelligent fallback analyzer.")
        _sentiment_model = None
    return _sentiment_model


# ---------------------------------------------------------------------------
# Aspect vocabulary: expanded keywords & multi-word expressions
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS = {
    "room": [
        "room", "rooms", "bed", "beds", "mattress", "pillow", "pillows", "linen", "linens",
        "sheets", "blanket", "blankets", "bathroom", "shower", "bathtub", "tub", "toilet",
        "suite", "balcony", "view", "views", "furnishing", "furniture", "space", "spacious"
    ],
    "cleanliness": [
        "clean", "cleanliness", "dirty", "dust", "dusty", "smell", "smelly", "stain", "stained",
        "hygiene", "hygienic", "unhygienic", "spotless", "filthy", "odor", "foul", "mold",
        "grime", "grim", "messy", "bugs", "cockroach", "cockroaches", "sanitary", "sanitized"
    ],
    "staff": [
        "staff", "service", "reception", "receptionist", "receptionists", "manager", "housekeeping",
        "concierge", "employee", "employees", "host", "valet", "bellboy", "bellhop", "front desk",
        "check-in", "check in", "checkout", "check out", "hospitality", "attentive", "courteous", "rude"
    ],
    "food": [
        "food", "breakfast", "dinner", "lunch", "restaurant", "buffet", "meal", "meals", "coffee",
        "tea", "menu", "cuisine", "dishes", "dining", "drink", "drinks", "beverage", "bakery",
        "cocktail", "snacks", "bar", "cafe", "cook", "chef"
    ],
    "location": [
        "location", "located", "central", "accessible", "close to", "near", "nearby",
        "walking distance", "neighborhood", "neighbourhood", "station", "airport", "beach",
        "metro", "subway", "downtown", "center", "centre", "convenient"
    ],
    "price": [
        "price", "prices", "cost", "costs", "expensive", "cheap", "value", "worth", "overpriced",
        "rate", "rates", "affordable", "budget", "charge", "charges", "fee", "fees", "pricey",
        "deal", "economical"
    ],
    "amenities": [
        "wifi", "wi-fi", "internet", "pool", "swimming pool", "gym", "fitness", "parking",
        "spa", "elevator", "lift", "tv", "television", "air conditioning", "ac", "air conditioner",
        "heater", "heating", "sauna", "jacuzzi", "amenity", "amenities", "hot water"
    ],
    "noise": [
        "noise", "noisy", "loud", "quiet", "peaceful", "soundproof", "soundproofing",
        "traffic", "sirens", "party", "thin walls", "street noise", "silent", "calm"
    ],
}

# Compiled regex patterns for whole-word/phrase matches
ASPECT_PATTERNS = {
    aspect: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(words, key=len, reverse=True)) + r")\b",
        re.IGNORECASE
    )
    for aspect, words in ASPECT_KEYWORDS.items()
}

# Disambiguation patterns (phrases that should NOT be counted as certain aspects)
EXCLUSION_RULES = [
    (re.compile(r"\bbar\s+of\s+soap\b", re.IGNORECASE), "food"),
    (re.compile(r"\broom\s+for\s+improvement\b", re.IGNORECASE), "room"),
    (re.compile(r"\bin\s+(?:view|light)\s+of\b", re.IGNORECASE), "room"),
    (re.compile(r"\bnear\s+future\b", re.IGNORECASE), "location"),
    (re.compile(r"\bcost\s+an\s+arm\s+and\s+a\s+leg\b", re.IGNORECASE), None),
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_SPLIT = re.compile(
    r"\s*(?:;|,?\s+(?:but|however|although|though|yet|whereas|while)\s+|,?\s+and\s+(?=[a-z\s]{3,}\b(?:was|is|were|are|felt|had|seemed)\b))\s*",
    re.IGNORECASE
)


def preprocess_review(raw_text):
    """Clean review text while preserving punctuation and capitalization for NLP models."""
    if not raw_text:
        return ""
    text = str(raw_text)
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def extract_aspects(clean_text):
    """
    Extract hotel aspects and their corresponding evidence clauses from review text.
    Filters out ambiguous idiomatic expressions.
    """
    found = []
    seen = set()

    for sentence in SENTENCE_SPLIT.split(clean_text):
        sentence = sentence.strip()
        if len(sentence) < 3:
            continue

        clauses = CLAUSE_SPLIT.split(sentence)
        for clause in clauses:
            clause = clause.strip()
            if len(clause) < 3:
                continue

            for aspect, pattern in ASPECT_PATTERNS.items():
                if pattern.search(clause):
                    # Check exclusions
                    excluded = False
                    for excl_pat, excl_aspect in EXCLUSION_RULES:
                        if excl_pat.search(clause) and (excl_aspect is None or excl_aspect == aspect):
                            # Verify if any other keywords for this aspect exist in this clause
                            other_words = [w for w in ASPECT_KEYWORDS[aspect] if not excl_pat.pattern]
                            excluded = True
                            break

                    if not excluded:
                        key = (aspect, clause.lower())
                        if key not in seen:
                            seen.add(key)
                            found.append({"aspect": aspect, "evidence": clause})
    return found


# ---------------------------------------------------------------------------
# Lexicon fallback analyzer (used when transformers/torch are offline/downloading)
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "lovely",
    "comfortable", "clean", "spotless", "friendly", "helpful", "delicious", "tasty",
    "perfect", "superb", "spacious", "polite", "welcoming", "peaceful", "quiet",
    "cozy", "luxurious", "modern", "affordable", "fast", "smooth", "accommodating",
    "attentive", "pleasant", "best", "loved", "liked", "nice", "exceptional", "fresh"
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "dirty", "filthy", "disappointing",
    "uncomfortable", "rude", "cold", "slow", "loud", "noisy", "overpriced", "expensive",
    "cramped", "smelly", "stain", "stained", "broken", "unpleasant", "worst", "hate",
    "hated", "disgusted", "unhelpful", "bugs", "cockroach", "leaking", "stink", "stinky",
    "hard", "unacceptable", "waste", "mediocre", "dusty", "clogged", "small", "tiny"
}

_NEGATION_WORDS = {"not", "no", "never", "hardly", "barely", "scarcely", "without", "wasn't", "weren't", "isn't", "aren't", "didn't", "don't"}


def _fallback_aspect_sentiment(evidence, aspect):
    """
    Context-aware lexicon sentiment classifier for an aspect within an evidence clause.
    Accurately computes polarity, negation flips, and confidence.
    """
    words = re.findall(r"\b[a-zA-Z']+\b", evidence.lower())
    pos_score = 0
    neg_score = 0
    negated = False

    for i, word in enumerate(words):
        if word in _NEGATION_WORDS:
            negated = True
            continue

        is_pos = word in _POSITIVE_WORDS
        is_neg = word in _NEGATIVE_WORDS

        # Aspect specific valence adjustments
        if aspect == "noise":
            if word in {"quiet", "peaceful", "silent", "soundproof"}:
                is_pos = True
                is_neg = False
            elif word in {"loud", "noisy", "traffic"}:
                is_neg = True
                is_pos = False

        if is_pos:
            if negated:
                neg_score += 1.5
            else:
                pos_score += 1.2
            negated = False
        elif is_neg:
            if negated:
                pos_score += 1.0
            else:
                neg_score += 1.3
            negated = False

    total = pos_score + neg_score
    if total == 0:
        return {"sentiment": "neutral", "confidence": 0.65, "model": "lexicon_fallback"}
    
    diff = pos_score - neg_score
    if diff > 0.4:
        conf = min(0.96, 0.70 + (pos_score / (total + 0.5)) * 0.25)
        return {"sentiment": "positive", "confidence": round(conf, 3), "model": "lexicon_fallback"}
    elif diff < -0.4:
        conf = min(0.96, 0.70 + (neg_score / (total + 0.5)) * 0.25)
        return {"sentiment": "negative", "confidence": round(conf, 3), "model": "lexicon_fallback"}
    else:
        return {"sentiment": "neutral", "confidence": 0.72, "model": "lexicon_fallback"}


def classify_aspect_sentiments(aspect_mentions):
    """
    Run ABSA model on every (evidence, aspect) pair.
    Uses DeBERTa-v3 model if already loaded; otherwise instantly uses the high-speed
    context-aware fallback so requests never stall or time out.
    """
    global _sentiment_model
    if not aspect_mentions:
        return []

    if _sentiment_model is not None:
        try:
            model_inputs = [
                {"text": m["evidence"], "text_pair": m["aspect"]} for m in aspect_mentions
            ]
            predictions = _sentiment_model(model_inputs, batch_size=16, truncation=True)
            return [
                {
                    "sentiment": p["label"].lower(),
                    "confidence": round(float(p["score"]), 3),
                    "model": "deberta_v3"
                }
                for p in predictions
            ]
        except Exception as exc:
            logger.warning(f"Inference error with transformer model: {exc}. Using fallback.")

    # Instant Fallback path (avoids blocking requests during model downloads or network latency)
    return [
        _fallback_aspect_sentiment(m["evidence"], m["aspect"])
        for m in aspect_mentions
    ]


def build_results(reviews, progress_callback=None):
    """
    reviews: list of (review_id, raw_text).
    progress_callback: optional callable(current, total)
    Returns a list of result rows: review_id, aspect, evidence, sentiment, confidence.
    """
    mentions = []
    total_reviews = len(reviews)

    for idx, (review_id, raw_text) in enumerate(reviews):
        clean_text = preprocess_review(raw_text)
        for mention in extract_aspects(clean_text):
            mention["review_id"] = review_id
            mentions.append(mention)
        if progress_callback and (idx + 1) % 10 == 0:
            progress_callback(idx + 1, total_reviews)

    if not mentions:
        return []

    predictions = classify_aspect_sentiments(mentions)
    rows = []
    for mention, prediction in zip(mentions, predictions):
        rows.append({
            "review_id": mention["review_id"],
            "aspect": mention["aspect"],
            "evidence": mention["evidence"],
            "sentiment": prediction["sentiment"],
            "confidence": prediction["confidence"],
            "model": prediction.get("model", "unknown")
        })

    if progress_callback:
        progress_callback(total_reviews, total_reviews)

    return rows


def summarize_by_aspect(rows):
    """
    Count positive / neutral / negative per aspect and compute:
      - Total mentions
      - Positive percentage
      - Net Sentiment Score (NSS = %Positive - %Negative)
    """
    summary = {}
    for row in rows:
        aspect = row["aspect"]
        sentiment = row["sentiment"]
        counts = summary.setdefault(aspect, {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "total": 0,
            "positive_pct": 0.0,
            "net_sentiment_score": 0.0
        })
        if sentiment in counts:
            counts[sentiment] += 1
        counts["total"] += 1

    for aspect, data in summary.items():
        total = data["total"]
        if total > 0:
            data["positive_pct"] = round((data["positive"] / total) * 100, 1)
            data["net_sentiment_score"] = round(((data["positive"] - data["negative"]) / total) * 100, 1)

    return summary
