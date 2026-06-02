import html
import io
import re
import textwrap
from urllib.parse import urlparse
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


st.set_page_config(
    page_title="Veritas AI | Fake News Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


PERFORMANCE_METRICS = {
    "Accuracy": "99.97%",
    "Precision": "99.98%",
    "Recall": "99.95%",
    "F1 Score": "99.96%",
}

DATASET_STATS = {
    "Total Articles": "44,898",
    "Training Samples": "35,918",
    "Testing Samples": "8,980",
    "Model": "DistilBERT",
}

EXAMPLE_INPUTS = {
    "Government Policy Update": (
        "The finance ministry announced a revised compliance framework for digital "
        "payments, requiring banks and fintech platforms to improve fraud monitoring "
        "and customer dispute resolution timelines."
    ),
    "Sensational Viral Claim": (
        "Scientists confirm that a newly discovered fruit can eliminate every known "
        "disease in 24 hours, according to a leaked international report."
    ),
    "Market and Technology News": (
        "A global technology company reported quarterly earnings above analyst "
        "expectations, citing stronger enterprise cloud adoption and disciplined "
        "cost management."
    ),
}

SUSPICIOUS_TERMS = {
    "shocking",
    "breaking",
    "miracle",
    "secret",
    "leaked",
    "exclusive",
    "guaranteed",
    "instant",
    "cure",
    "unbelievable",
    "viral",
    "urgent",
    "scandal",
    "hoax",
    "conspiracy",
    "exposed",
    "banned",
    "anonymous",
    "sensational",
    "clickbait",
    "alarming",
}

SOURCE_CREDIBILITY = {
    "reuters": (95, "Very High", "Reuters is widely regarded as a high-reliability wire service with strong editorial standards."),
    "associated press": (94, "Very High", "The Associated Press (AP) is known for rigorous reporting standards and broad newsroom review."),
    "ap": (94, "Very High", "The Associated Press (AP) is known for rigorous reporting standards and broad newsroom review."),
    "bbc": (92, "High", "BBC is generally considered a high-credibility outlet with established editorial governance."),
    "the guardian": (86, "High", "The Guardian is a major outlet with editorial oversight, though coverage may vary by topic."),
    "nytimes": (88, "High", "The New York Times typically has strong editorial processes and corrections policies."),
    "new york times": (88, "High", "The New York Times typically has strong editorial processes and corrections policies."),
    "cnn": (84, "High", "CNN is a major outlet; credibility can vary by segment, but it generally follows newsroom standards."),
    "al jazeera": (82, "Medium", "Al Jazeera is a large network; credibility can vary by region and topic."),
    "blog": (58, "Low", "Unverified blogs can vary significantly in quality. Cross-check claims with reputable sources."),
    "facebook": (52, "Low", "Social posts may be unverified. Validate with trusted outlets and original sources."),
    "twitter": (52, "Low", "Social posts may be unverified. Validate with trusted outlets and original sources."),
    "x": (52, "Low", "Social posts may be unverified. Validate with trusted outlets and original sources."),
}

SENTIMENT_POSITIVE = {
    "growth",
    "strong",
    "improve",
    "improved",
    "success",
    "successful",
    "positive",
    "benefit",
    "benefits",
    "stable",
    "stability",
    "record",
    "wins",
    "win",
    "gain",
    "gains",
    "increase",
    "increased",
    "innovation",
    "secure",
    "secured",
    "safe",
    "efficient",
    "leading",
}

SENTIMENT_NEGATIVE = {
    "crisis",
    "fraud",
    "scam",
    "fake",
    "hoax",
    "collapse",
    "panic",
    "fear",
    "risk",
    "threat",
    "threats",
    "warning",
    "lawsuit",
    "illegal",
    "ban",
    "banned",
    "decline",
    "declined",
    "loss",
    "losses",
    "negative",
    "attack",
    "attacks",
}


@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("saved_model")
    model = DistilBertForSequenceClassification.from_pretrained("saved_model")
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


if "news_input" not in st.session_state:
    st.session_state.news_input = ""

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "prediction_history" not in st.session_state:
    # Stores recent predictions for the current session only (Streamlit session state).
    st.session_state.prediction_history = []

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"


def run_inference(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probabilities = torch.softmax(logits, dim=0)
    prediction = torch.argmax(probabilities).item()

    fake_probability = float(probabilities[0].item() * 100)
    real_probability = float(probabilities[1].item() * 100)
    confidence = max(fake_probability, real_probability)

    return {
        "prediction": prediction,
        "label": "Real News" if prediction == 1 else "Fake News",
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "probability_margin": abs(real_probability - fake_probability),
    }


def append_prediction_history(result: dict) -> None:
    st.session_state.prediction_history.append(
        {
            "Timestamp (UTC)": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Prediction": result["label"],
            "Confidence (%)": round(float(result["confidence"]), 2),
            "Sentiment": st.session_state.get("latest_sentiment", {}).get("label", "—"),
        }
    )
    # Keep only the most recent 20 entries for a clean recruiter-friendly table.
    st.session_state.prediction_history = st.session_state.prediction_history[-20:]

def analyze_sentiment(text: str) -> dict:
    tokens = [t.strip(".,!?;:()[]{}\"'").lower() for t in text.split()]
    tokens = [t for t in tokens if t]

    pos = sum(1 for t in tokens if t in SENTIMENT_POSITIVE)
    neg = sum(1 for t in tokens if t in SENTIMENT_NEGATIVE)
    total = pos + neg

    if total == 0:
        return {
            "label": "Neutral",
            "confidence": 55.0,
            "summary": "The text does not contain strong positive/negative lexical signals; sentiment is likely neutral.",
        }

    score = (pos - neg) / total
    confidence = min(95.0, 55.0 + abs(score) * 40.0 + (min(total, 10) * 2.0))

    if score >= 0.25:
        label = "Positive"
        summary = "Overall tone contains more positive cues than negative cues."
    elif score <= -0.25:
        label = "Negative"
        summary = "Overall tone contains more negative cues than positive cues."
    else:
        label = "Neutral"
        summary = "Tone is mixed with no dominant positive/negative direction."

    return {"label": label, "confidence": round(confidence, 2), "summary": summary}


def assess_source_credibility(source_name: str) -> dict:
    cleaned = (source_name or "").strip().lower()
    if not cleaned:
        return {
            "score": None,
            "tier": "Not provided",
            "explanation": "Optional: enter a publisher or source name to estimate credibility context.",
        }

    for key, (score, tier, explanation) in SOURCE_CREDIBILITY.items():
        if cleaned == key or key in cleaned:
            return {"score": score, "tier": tier, "explanation": explanation}

    # Fallback heuristic for unknown sources.
    if "blog" in cleaned or "forum" in cleaned:
        return {
            "score": 58,
            "tier": "Low",
            "explanation": "Independent blogs/forums may have limited editorial oversight. Verify claims using primary sources.",
        }

    return {
        "score": 72,
        "tier": "Medium",
        "explanation": "Source is not in the preset list. Treat as medium credibility and validate with multiple trusted outlets.",
    }


def get_risk_profile(prediction: int, fake_probability: float) -> tuple[str, str]:
    if prediction == 0 and fake_probability >= 85:
        return "High Risk", "Substantial misinformation risk detected."
    if prediction == 0 and fake_probability >= 65:
        return "Elevated Risk", "Signals indicate notable credibility concerns."
    if prediction == 1 and fake_probability <= 25:
        return "Low Risk", "The content appears structurally consistent with credible reporting."
    return "Moderate Risk", "The prediction is meaningful, but the content still warrants source verification."


def get_confidence_note(confidence: float) -> str:
    if confidence >= 90:
        return "High model certainty based on strongly separated class probabilities."
    if confidence >= 75:
        return "Solid confidence with a clear decision boundary between classes."
    return "Moderate confidence; external fact-checking is recommended for final validation."


def build_probability_chart(real_probability: float, fake_probability: float):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Real", "Fake"],
                values=[real_probability, fake_probability],
                hole=0.72,
                sort=False,
                marker={
                    "colors": ["#4f8cff", "#f97316"],
                    "line": {"color": "rgba(15, 23, 42, 0.95)", "width": 2},
                },
                textinfo="label+percent",
                textfont={"color": "#e5eefb", "size": 14},
                hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="#e5eefb", family="Inter, sans-serif"),
    )
    return fig


def compute_trust_score(result: dict) -> float:
    """
    Trust Score (0-100):
    - If prediction = REAL: Trust Score = Real Probability
    - If prediction = FAKE: Trust Score = 100 - Fake Probability
    """
    if result.get("prediction") == 1:
        return float(result.get("real_probability", 0.0))
    return float(100.0 - result.get("fake_probability", 0.0))


def confidence_level_label(confidence: float) -> str:
    if confidence >= 90:
        return "High"
    if confidence >= 75:
        return "Medium"
    return "Moderate"


def build_trust_score_gauge(trust_score: float, theme_mode: str):
    bar_color = "#4f8cff" if theme_mode == "Dark" else "#2563eb"
    text_color = "#e5eefb" if theme_mode == "Dark" else "#0f172a"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(trust_score),
            number={"suffix": "%", "font": {"size": 38, "color": text_color}},
            title={"text": "Trust Score", "font": {"size": 16, "color": text_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(148,163,184,0.35)"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.25)"},
                    {"range": [40, 70], "color": "rgba(249, 115, 22, 0.25)"},
                    {"range": [70, 100], "color": "rgba(34, 197, 94, 0.20)"},
                ],
                "threshold": {"line": {"color": "rgba(148,163,184,0.55)", "width": 2}, "value": float(trust_score)},
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color, family="Inter, sans-serif"),
        transition={"duration": 450},
    )
    return fig


def extract_article(url: str) -> dict:
    """
    Extract article metadata + text using newspaper3k (preferred) or a lightweight fallback.
    Returns:
      { ok: bool, title, authors, publish_date, text, error }
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "Please enter a valid URL."}

    # Preferred: newspaper3k
    try:
        from newspaper import Article  # type: ignore

        article = Article(url)
        article.download()
        article.parse()
        return {
            "ok": True,
            "title": getattr(article, "title", "") or "",
            "authors": getattr(article, "authors", []) or [],
            "publish_date": getattr(article, "publish_date", None),
            "text": getattr(article, "text", "") or "",
        }
    except Exception:
        pass

    # Fallback: requests + simple paragraph extraction
    if requests is None:
        return {
            "ok": False,
            "error": "URL extraction requires 'newspaper3k' (recommended) or 'requests'. Please install newspaper3k.",
        }

    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html_text = resp.text
        # Very lightweight text extraction: grab paragraph content.
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = []
        for p in paragraphs:
            p = re.sub(r"<[^>]+>", " ", p)
            p = html.unescape(p)
            p = re.sub(r"\s+", " ", p).strip()
            if len(p) >= 40:
                cleaned.append(p)
        text = "\n\n".join(cleaned).strip()
        if not text:
            return {
                "ok": False,
                "error": "Unable to extract article content from this URL. Try installing newspaper3k or paste the article text directly.",
            }
        return {
            "ok": True,
            "title": "",
            "authors": [],
            "publish_date": None,
            "text": text,
        }
    except Exception as exc:
        return {"ok": False, "error": f"Article extraction failed: {exc}"}


def guess_source_from_url(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""
    if "bbc." in domain:
        return "BBC"
    if "reuters." in domain:
        return "Reuters"
    if "cnn." in domain:
        return "CNN"
    if "theguardian." in domain or "guardian." in domain:
        return "The Guardian"
    if "nytimes." in domain:
        return "New York Times"
    return domain.split(".")[0].title() if domain else ""


def detect_text_columns(df: pd.DataFrame) -> list[str]:
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        return []

    scored = []
    for c in object_cols:
        series = df[c].dropna().astype(str)
        if series.empty:
            continue
        avg_len = float(series.str.len().mean())
        scored.append((c, avg_len))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


def run_batch_analysis(df: pd.DataFrame, text_col: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    total = int(len(df))
    for idx, raw_text in enumerate(df[text_col].fillna("").astype(str).tolist()):
        text = raw_text.strip()
        if not text:
            rows.append(
                {
                    "Prediction": "—",
                    "Confidence Score": 0.0,
                    "Real Probability": 0.0,
                    "Fake Probability": 0.0,
                    "Sentiment": "—",
                    "Risk Level": "—",
                }
            )
            continue

        result = run_inference(text)
        sentiment = analyze_sentiment(text)
        risk_level, _ = get_risk_profile(result["prediction"], result["fake_probability"])

        rows.append(
            {
                "Prediction": result["label"],
                "Confidence Score": round(float(result["confidence"]), 2),
                "Real Probability": round(float(result["real_probability"]), 2),
                "Fake Probability": round(float(result["fake_probability"]), 2),
                "Sentiment": sentiment["label"],
                "Risk Level": risk_level,
            }
        )

    out = df.copy()
    for key in rows[0].keys() if rows else []:
        out[key] = [r[key] for r in rows]

    summary = {
        "total": total,
        "real_count": int((out["Prediction"] == "Real News").sum()) if "Prediction" in out else 0,
        "fake_count": int((out["Prediction"] == "Fake News").sum()) if "Prediction" in out else 0,
        "avg_confidence": float(out["Confidence Score"].mean()) if "Confidence Score" in out and total else 0.0,
    }
    return out, summary

def build_reason_signals(text: str, result: dict) -> list[str]:
    cleaned_words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
    word_count = len([word for word in cleaned_words if word])
    detected_terms = sorted({word for word in cleaned_words if word in SUSPICIOUS_TERMS})

    uppercase_ratio = (
        sum(1 for char in text if char.isupper()) / max(sum(1 for char in text if char.isalpha()), 1)
    )
    punctuation_intensity = text.count("!") + text.count("?")

    reasons = [
        f"The model assigned {result['confidence']:.2f}% confidence with a {result['probability_margin']:.2f} point margin between the two classes.",
        (
            f"The submitted text contains {word_count} words, which provides a "
            f"{'substantial' if word_count >= 35 else 'limited'} context window for classification."
        ),
    ]

    if detected_terms:
        reasons.append(
            "Detected attention-oriented language cues: "
            + ", ".join(detected_terms[:5])
            + ". These cues often correlate with lower content credibility."
        )
    elif uppercase_ratio > 0.18 or punctuation_intensity >= 3:
        reasons.append(
            "Stylistic emphasis (capitalization and punctuation intensity) is elevated, which can influence perceived credibility."
        )
    else:
        reasons.append(
            "The language structure appears relatively neutral, so the prediction is driven more by semantic context than sensational phrasing."
        )

    reasons.append(
        "This explainability panel is advisory only; the final classification is derived directly from DistilBERT logits without post-processing manipulation."
    )
    return reasons


def aggregate_word_importance(input_ids: torch.Tensor, saliency_scores: torch.Tensor) -> list[dict]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    merged_words = []
    current_word = ""
    current_scores = []

    for token, score in zip(tokens, saliency_scores.tolist()):
        if token in tokenizer.all_special_tokens:
            continue

        clean_token = token.replace("▁", "").replace("Ġ", "")
        if token.startswith("##"):
            current_word += clean_token.replace("##", "")
            current_scores.append(float(score))
            continue

        if current_word:
            normalized = re.sub(r"[^A-Za-z0-9'-]", "", current_word).strip("-'")
            if normalized:
                merged_words.append((normalized, max(current_scores)))

        current_word = clean_token.replace("##", "")
        current_scores = [float(score)]

    if current_word:
        normalized = re.sub(r"[^A-Za-z0-9'-]", "", current_word).strip("-'")
        if normalized:
            merged_words.append((normalized, max(current_scores)))

    deduplicated = {}
    for word, score in merged_words:
        lower_word = word.lower()
        if len(lower_word) <= 2:
            continue
        deduplicated[lower_word] = max(deduplicated.get(lower_word, 0.0), score)

    if not deduplicated:
        return []

    max_score = max(deduplicated.values()) or 1.0
    ranked_words = sorted(deduplicated.items(), key=lambda item: item[1], reverse=True)[:8]

    return [
        {"word": word, "score": round((score / max_score) * 100, 2)}
        for word, score in ranked_words
    ]


def compute_xai_insights(text: str, result: dict) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )

    model.zero_grad(set_to_none=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embeddings = model.distilbert.embeddings(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)
    embeddings.retain_grad()

    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    predicted_class = int(torch.argmax(outputs.logits, dim=1).item())
    target_logit = outputs.logits[0, predicted_class]
    target_logit.backward()

    token_scores = embeddings.grad.norm(dim=2).squeeze(0).detach().cpu()
    influential_words = aggregate_word_importance(input_ids.squeeze(0).detach().cpu(), token_scores)

    suspicious_words = sorted(
        {
            item["word"]
            for item in influential_words
            if item["word"].lower() in SUSPICIOUS_TERMS
        }
    )

    if not suspicious_words:
        suspicious_words = sorted(
            {
                word.strip(".,!?;:()[]{}\"'").lower()
                for word in text.split()
                if word.strip(".,!?;:()[]{}\"'").lower() in SUSPICIOUS_TERMS
            }
        )[:8]

    return {
        "influential_words": influential_words,
        "suspicious_words": suspicious_words,
        "explanation": build_prediction_explanation(result, influential_words, suspicious_words),
        "highlighted_text": build_highlighted_text(text, influential_words, suspicious_words),
    }


def build_prediction_explanation(
    result: dict, influential_words: list[dict], suspicious_words: list[str]
) -> dict:
    top_terms = [item["word"] for item in influential_words[:5]]
    influential_text = ", ".join(top_terms) if top_terms else "contextual language patterns"

    if result["prediction"] == 0:
        title = "Why the model predicted Fake News"
        details = (
            f"The fake-news class received the stronger probability signal "
            f"({result['fake_probability']:.2f}% vs {result['real_probability']:.2f}%). "
            f"The most influential terms included {influential_text}."
        )
        if suspicious_words:
            details += (
                " The text also contains suspicious lexical cues such as "
                + ", ".join(suspicious_words[:5])
                + ", which reinforce misinformation-style patterns."
            )
        else:
            details += " The strongest evidence came from semantic context rather than explicit suspicious keywords."
    else:
        title = "Why the model predicted Real News"
        details = (
            f"The real-news class received the stronger probability signal "
            f"({result['real_probability']:.2f}% vs {result['fake_probability']:.2f}%). "
            f"The most influential terms included {influential_text}, suggesting a more structured reporting pattern."
        )
        if suspicious_words:
            details += (
                " A few attention-oriented terms were detected, but they were not strong enough to overturn the dominant real-news signal."
            )
        else:
            details += " The text does not rely heavily on suspicious trigger words among the highest-impact terms."

    return {"title": title, "details": details}


def build_highlighted_text(
    original_text: str, influential_words: list[dict], suspicious_words: list[str]
) -> str:
    influential_set = {item["word"].lower() for item in influential_words[:8]}
    suspicious_set = {word.lower() for word in suspicious_words}

    parts = re.findall(r"\s+|[A-Za-z0-9'-]+|[^\w\s]", original_text)
    highlighted = []

    for part in parts:
        if part.isspace():
            highlighted.append(part.replace("\n", "<br>"))
            continue

        normalized = re.sub(r"[^A-Za-z0-9'-]", "", part).lower()
        safe_part = html.escape(part)

        if normalized and normalized in suspicious_set:
            highlighted.append(f"<span class='highlight-token suspicious-token'>{safe_part}</span>")
        elif normalized and normalized in influential_set:
            highlighted.append(f"<span class='highlight-token influence-token'>{safe_part}</span>")
        else:
            highlighted.append(safe_part)

    return "".join(highlighted)


def build_influential_word_chips(words: list[dict]) -> str:
    if not words:
        return "<span class='chip-muted'>No high-impact tokens were available for display.</span>"

    chips = []
    for item in words:
        chips.append(
            f"<span class='word-chip'>{html.escape(item['word'])}<span class='chip-score'>{item['score']:.0f}%</span></span>"
        )
    return "".join(chips)


def build_suspicious_word_chips(words: list[str]) -> str:
    if not words:
        return "<span class='chip-muted'>No suspicious keywords were detected in the submitted text.</span>"

    chips = []
    for word in words:
        chips.append(f"<span class='word-chip suspicious-chip'>{html.escape(word)}</span>")
    return "".join(chips)


def escape_pdf_text(value: str) -> str:
    safe_value = value.encode("latin-1", "replace").decode("latin-1")
    return safe_value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_simple_pdf(lines: list[str]) -> bytes:
    page_height = 792
    start_x = 50
    start_y = 750
    line_gap = 16

    content_ops = ["BT", "/F1 11 Tf", f"{start_x} {start_y} Td"]
    first_line = True

    for raw_line in lines:
        if not raw_line:
            raw_line = " "
        for wrapped in textwrap.wrap(raw_line, width=92) or [" "]:
            safe_line = escape_pdf_text(wrapped)
            if first_line:
                content_ops.append(f"({safe_line}) Tj")
                first_line = False
            else:
                content_ops.append(f"0 -{line_gap} Td")
                content_ops.append(f"({safe_line}) Tj")

            current_line_index = len([op for op in content_ops if op.endswith("Tj")])
            current_y = start_y - ((current_line_index - 1) * line_gap)
            if current_y <= 60:
                break

        current_line_index = len([op for op in content_ops if op.endswith("Tj")])
        current_y = start_y - ((current_line_index - 1) * line_gap)
        if current_y <= 60:
            break

    content_ops.append("ET")
    content_stream = "\n".join(content_ops).encode("latin-1", "replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    objects.append(
        f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 {page_height}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj".encode("latin-1")
    )
    objects.append(
        f"4 0 obj << /Length {len(content_stream)} >> stream\n".encode("latin-1")
        + content_stream
        + b"\nendstream endobj"
    )
    objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
        pdf.extend(b"\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)


def generate_report_bytes(
    source_text: str,
    result: dict,
    explainability: list[str],
    xai_data: dict,
) -> bytes:
    risk_label, risk_note = get_risk_profile(result["prediction"], result["fake_probability"])
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    influential_summary = ", ".join(item["word"] for item in xai_data.get("influential_words", [])[:6])
    suspicious_summary = ", ".join(xai_data.get("suspicious_words", [])[:6]) or "None detected"

    report_lines = [
        "Fake News Detection Analysis Report",
        f"Generated: {timestamp}",
        "",
        "Prediction Summary",
        f"Classification: {result['label']}",
        f"Confidence Score: {result['confidence']:.2f}%",
        f"Real Probability: {result['real_probability']:.2f}%",
        f"Fake Probability: {result['fake_probability']:.2f}%",
        f"Risk Indicator: {risk_label}",
        f"Risk Note: {risk_note}",
        "",
        "Input Text Preview",
        source_text,
        "",
        "Explainability Notes",
    ]
    report_lines.extend(f"- {item}" for item in explainability)
    report_lines.extend(
        [
            "",
            "Explainable AI Summary",
            f"- Explanation Card: {xai_data.get('explanation', {}).get('details', '')}",
            f"- Most Influential Words: {influential_summary or 'Not available'}",
            f"- Suspicious Words: {suspicious_summary}",
            "",
            "Model Information",
            "- Backbone: DistilBERT",
            "- Framework: PyTorch + Hugging Face Transformers",
            "- Inference Path: Tokenizer -> DistilBERT -> Classification Layer -> Prediction",
        ]
    )
    return build_simple_pdf(report_lines)


st.markdown(
    """
    <style>
    :root {
        --bg-primary: #0b1020;
        --bg-secondary: rgba(15, 23, 42, 0.74);
        --border-soft: rgba(148, 163, 184, 0.16);
        --text-primary: #f8fafc;
        --text-secondary: #a8b3cf;
        --blue: #4f8cff;
        --cyan: #67e8f9;
        --orange: #f97316;
        --green: #22c55e;
        --warning: #fb923c;
        --soft-surface: rgba(13, 18, 34, 0.7);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(79, 140, 255, 0.16), transparent 22%),
            radial-gradient(circle at top right, rgba(103, 232, 249, 0.10), transparent 18%),
            linear-gradient(180deg, #0b1020 0%, #0f172a 45%, #101826 100%);
        color: var(--text-primary);
    }

    .main .block-container {
        max-width: 1240px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(9, 14, 28, 0.92);
        border-right: 1px solid var(--border-soft);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(12, 19, 36, 0.82) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 18px !important;
        min-height: 300px !important;
        line-height: 1.65 !important;
        padding: 1rem 1rem 1.1rem 1rem !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        min-height: 2.95rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Premium primary action button (Analyze Content) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #4f46e5, #7c3aed) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 6px 22px rgba(79, 70, 229, 0.28);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.40);
        filter: brightness(1.03);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: var(--soft-surface);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 16px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(148, 163, 184, 0.22);
        box-shadow: 0 22px 60px rgba(2, 6, 23, 0.22);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        padding: 34px;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(17, 24, 39, 0.72));
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 20px 60px rgba(2, 6, 23, 0.28);
        margin-bottom: 18px;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        top: -120px;
        right: -40px;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(79, 140, 255, 0.24), transparent 70%);
        pointer-events: none;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(79, 140, 255, 0.10);
        border: 1px solid rgba(79, 140, 255, 0.18);
        color: #d7e7ff;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: clamp(2.2rem, 5vw, 3.6rem);
        line-height: 1.05;
        font-weight: 750;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        margin: 0 0 10px 0;
    }

    .hero-copy {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.75;
        max-width: 760px;
        margin: 0;
    }

    .surface-card {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid var(--border-soft);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.20);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .surface-card + .surface-card {
        margin-top: 16px;
    }

    .surface-card:hover {
        transform: translateY(-2px);
        border-color: rgba(148, 163, 184, 0.22);
        box-shadow: 0 22px 60px rgba(2, 6, 23, 0.22);
    }

    .section-title {
        color: var(--text-primary);
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .section-copy {
        color: var(--text-secondary);
        margin-bottom: 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    .stat-card {
        padding: 18px;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.72), rgba(15, 23, 42, 0.72));
        border: 1px solid var(--border-soft);
        height: 100%;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(148, 163, 184, 0.24);
        box-shadow: 0 22px 60px rgba(2, 6, 23, 0.26);
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    .stat-value {
        color: var(--text-primary);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }

    .stat-foot {
        color: #7dd3fc;
        font-size: 0.84rem;
    }

    .result-banner {
        padding: 20px 22px;
        border-radius: 20px;
        margin-bottom: 16px;
        border: 1px solid var(--border-soft);
    }

    .result-real {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.14), rgba(15, 23, 42, 0.85));
        border-color: rgba(34, 197, 94, 0.24);
    }

    .result-fake {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(15, 23, 42, 0.85));
        border-color: rgba(249, 115, 22, 0.28);
    }

    .result-title {
        font-size: 1.24rem;
        font-weight: 750;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .result-copy {
        color: var(--text-secondary);
        margin-bottom: 0;
        line-height: 1.6;
    }

    .progress-shell {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.14);
        overflow: hidden;
        margin-top: 8px;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #60a5fa, #4f8cff, #67e8f9);
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid var(--border-soft);
        background: rgba(15, 23, 42, 0.6);
        color: var(--text-primary);
    }

    .explain-list {
        margin: 0;
        padding-left: 18px;
        color: var(--text-secondary);
        line-height: 1.85;
    }

    .arch-grid {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
    }

    .arch-step {
        padding: 12px 16px;
        border-radius: 14px;
        background: rgba(12, 19, 36, 0.86);
        border: 1px solid var(--border-soft);
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.92rem;
    }

    .arch-arrow {
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        font-size: 1rem;
    }

    .subtle-note {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.7;
    }

    .word-bank {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
    }

    .word-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 12px;
        border-radius: 999px;
        background: rgba(79, 140, 255, 0.10);
        border: 1px solid rgba(79, 140, 255, 0.18);
        color: var(--text-primary);
        font-size: 0.84rem;
        font-weight: 600;
    }

    .chip-score {
        color: #7dd3fc;
        font-size: 0.75rem;
    }

    .suspicious-chip {
        background: rgba(249, 115, 22, 0.12);
        border-color: rgba(249, 115, 22, 0.25);
    }

    .chip-muted {
        color: var(--text-secondary);
        font-size: 0.92rem;
    }

    .highlight-box {
        background: rgba(11, 16, 32, 0.82);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 18px;
        color: var(--text-secondary);
        line-height: 1.85;
        font-size: 0.95rem;
    }

    .highlight-token {
        display: inline-block;
        padding: 2px 8px;
        margin: 0 1px;
        border-radius: 8px;
        color: var(--text-primary);
    }

    .influence-token {
        background: rgba(79, 140, 255, 0.16);
        border: 1px solid rgba(79, 140, 255, 0.22);
    }

    .suspicious-token {
        background: rgba(249, 115, 22, 0.18);
        border: 1px solid rgba(249, 115, 22, 0.28);
    }

    .explanation-panel {
        padding: 18px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(79, 140, 255, 0.10), rgba(15, 23, 42, 0.88));
        border: 1px solid rgba(79, 140, 255, 0.20);
    }

    .explanation-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 8px;
    }

    .explanation-copy {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.75;
        margin-bottom: 0;
    }

    @media (max-width: 768px) {
        .hero-shell {
            padding: 24px;
            border-radius: 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.theme_mode == "Light":
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #f8fafc;
            --bg-secondary: rgba(255, 255, 255, 0.86);
            --border-soft: rgba(15, 23, 42, 0.12);
            --text-primary: #0f172a;
            --text-secondary: rgba(15, 23, 42, 0.72);
            --soft-surface: rgba(255, 255, 255, 0.82);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 140, 255, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(103, 232, 249, 0.10), transparent 22%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 45%, #ffffff 100%);
            color: var(--text-primary);
        }

        [data-testid="stSidebar"] {
            background: rgba(248, 250, 252, 0.95);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }

        div[data-testid="stTextArea"] textarea {
            background: rgba(255, 255, 255, 0.95) !important;
        }

        .surface-card,
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.86);
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(238, 242, 255, 0.86));
        }

        .result-real {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.14), rgba(255, 255, 255, 0.92));
        }

        .result-fake {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(255, 255, 255, 0.92));
        }

        .highlight-box {
            background: rgba(255, 255, 255, 0.88);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## 🛡️ Veritas AI")
    st.caption("Fake News Intelligence Platform")
    st.markdown("---")

    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        horizontal=True,
    )
    if theme_choice != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_choice
        st.rerun()

    st.markdown("### Model Information")
    st.markdown(
        """
        - **Backbone:** DistilBERT  
        - **Framework:** PyTorch  
        - **Library:** Hugging Face Transformers  
        - **Inference Mode:** Sequence classification  
        """
    )

    st.markdown("### Dataset Statistics")
    st.markdown(
        """
        - **Total Articles:** 44,898  
        - **Training Samples:** 35,918  
        - **Testing Samples:** 8,980  
        """
    )

    st.markdown("### Application Information")
    st.markdown(
        """
        - Production-style Streamlit interface  
        - Real-time DistilBERT inference  
        - PDF analysis export  
        - Probability visualization with Plotly  
        - Explainable AI word attribution  
        """
    )

    st.markdown("---")
    st.caption("Designed for portfolio presentation, recruiter review, and AI product showcase.")


st.markdown(
    """
    <div class="hero-shell">
        <div class="eyebrow">Fake News Intelligence Platform</div>
        <h1 class="hero-title">Veritas AI</h1>
        <p class="hero-copy">
            Powered by DistilBERT • Explainable AI • Real-Time Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">Model Performance Dashboard</div>
        <p class="section-copy">
            Reference evaluation metrics from the trained classification pipeline presented in a recruiter-friendly product format.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_footers = {
    "Accuracy": "Overall evaluation performance",
    "Precision": "False positive control",
    "Recall": "Detection coverage",
    "F1 Score": "Balanced effectiveness",
}

for column, (label, value) in zip(metric_columns, PERFORMANCE_METRICS.items()):
    with column:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-foot">{metric_footers[label]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">Dataset Information</div>
        <p class="section-copy">
            Training and evaluation context that communicates data scale, model type, and readiness for real-world analysis workflows.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

dataset_columns = st.columns(4)
dataset_footers = {
    "Total Articles": "Full labeled corpus",
    "Training Samples": "Used for model learning",
    "Testing Samples": "Held-out evaluation set",
    "Model": "Transformer backbone",
}

for column, (label, value) in zip(dataset_columns, DATASET_STATS.items()):
    with column:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-foot">{dataset_footers[label]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">Business Impact</div>
        <p class="section-copy">
            A startup-grade feature set designed to support real-world credibility checks and stakeholder-friendly reporting.
        </p>
        <div style="height: 12px;"></div>
        <ul class="explain-list">
            <li>✔ Real-Time News Verification</li>
            <li>✔ Explainable AI Decision Support</li>
            <li>✔ Credibility Assessment</li>
            <li>✔ Misinformation Risk Detection</li>
            <li>✔ Professional PDF Reporting</li>
            <li>✔ Production Ready Streamlit Deployment</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">Why This Project Matters</div>
        <p class="section-copy">
            This system demonstrates end-to-end AI product engineering through:
        </p>
        <div style="height: 10px;"></div>
        <ul class="explain-list">
            <li>NLP</li>
            <li>Transformers</li>
            <li>Explainable AI</li>
            <li>Product Design</li>
            <li>Data Visualization</li>
            <li>AI Deployment</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

workspace_left, workspace_right = st.columns([1.65, 1], gap="large")

with workspace_left:
    st.markdown(
        """
        <div class="surface-card">
            <div class="section-title">News Analysis Workspace</div>
            <p class="section-copy">
                Submit a headline, article excerpt, or social post for real-time classification. The inference pipeline remains unchanged and uses direct DistilBERT outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    example_columns = st.columns(3)
    for column, (title, sample_text) in zip(example_columns, EXAMPLE_INPUTS.items()):
        with column:
            if st.button(title, use_container_width=True):
                st.session_state.news_input = sample_text

    st.session_state.source_name = st.text_input(
        "News source (optional)",
        value=st.session_state.source_name,
        placeholder="e.g., Reuters, BBC, CNN, Blog",
    )

    news_text = st.text_area(
        "Paste article or headline",
        key="news_input",
        placeholder="Enter the news content you want to analyze...",
    )

    button_columns = st.columns([1.2, 1, 1.2])
    with button_columns[0]:
        analyze_clicked = st.button("Analyze Content", type="primary", use_container_width=True)
    with button_columns[1]:
        clear_clicked = st.button("Clear", use_container_width=True)
    with button_columns[2]:
        st.markdown(
            "<div class='subtle-note' style='padding-top: 9px;'>Example inputs help demonstrate the dashboard during portfolio reviews.</div>",
            unsafe_allow_html=True,
        )

    if clear_clicked:
        st.session_state.news_input = ""
        st.rerun()

with workspace_right:
    st.markdown(
        """
        <div class="surface-card">
            <div class="section-title">About the Model</div>
            <p class="section-copy">
                This application uses a DistilBERT sequence classifier running on PyTorch with Hugging Face Transformers. The interface emphasizes deployment quality, clarity, and recruiter appeal.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface-card">
            <div class="section-title">Architecture</div>
            <div class="arch-grid">
                <div class="arch-step">User Input</div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">Tokenizer</div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">DistilBERT</div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">Classification Layer</div>
                <div class="arch-arrow">→</div>
                <div class="arch-step">Prediction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface-card">
            <div class="section-title">Application Fit</div>
            <p class="section-copy">
                Positioned as a SaaS-style AI product dashboard, this interface demonstrates practical UX thinking, analytics presentation, and production-oriented ML integration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


analysis_result = None
explainability_items = []
xai_data = {}

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">Analyze News URL</div>
        <p class="section-copy">
            Paste a news article URL to extract content automatically and run the full Veritas AI analysis pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

url_cols = st.columns([2.2, 0.8])
with url_cols[0]:
    news_url = st.text_input(
        "News article URL",
        key="news_url",
        placeholder="https://example.com/news/article",
        label_visibility="collapsed",
    )
with url_cols[1]:
    url_analyze_clicked = st.button("Extract & Analyze", use_container_width=True)

if url_analyze_clicked:
    if not (news_url or "").strip():
        st.warning("Please paste a valid news URL.")
    else:
        with st.spinner("Extracting article content..."):
            extracted = extract_article(news_url)

        if not extracted.get("ok"):
            st.warning(extracted.get("error", "Unable to extract article content from this URL."))
        else:
            extracted_text = (extracted.get("text") or "").strip()
            if not extracted_text:
                st.warning("Article text was empty after extraction. Try another URL or paste the text manually.")
            else:
                # Pre-fill source name from the domain to keep the credibility system consistent.
                guessed_source = guess_source_from_url(news_url)
                if guessed_source:
                    st.session_state.source_name = guessed_source

                with st.expander("Extracted Article Preview", expanded=False):
                    st.markdown(f"**Title:** {extracted.get('title') or '—'}")
                    authors = extracted.get("authors") or []
                    st.markdown(f"**Author:** {', '.join(authors) if authors else '—'}")
                    pub = extracted.get("publish_date")
                    pub_value = pub.strftime("%Y-%m-%d") if hasattr(pub, "strftime") else "—"
                    st.markdown(f"**Publish Date:** {pub_value}")
                    st.text_area(
                        "Extracted text (preview)",
                        value=extracted_text[:3000],
                        height=220,
                    )

                # Optionally copy extracted text into the main workspace for transparency.
                st.session_state.news_input = extracted_text

                with st.spinner("Running DistilBERT inference..."):
                    analysis_result = run_inference(extracted_text)
                    sentiment_result = analyze_sentiment(extracted_text)
                    source_result = assess_source_credibility(st.session_state.source_name)
                    explainability_items = build_reason_signals(extracted_text, analysis_result)
                    xai_data = compute_xai_insights(extracted_text, analysis_result)

                    # Keep history intact but ensure this run has sentiment/source available in state.
                    st.session_state["latest_sentiment"] = sentiment_result
                    st.session_state["latest_source"] = source_result
                    append_prediction_history(analysis_result)

                st.session_state["latest_result"] = analysis_result
                st.session_state["latest_text"] = extracted_text
                st.session_state["latest_explainability"] = explainability_items
                st.session_state["latest_xai"] = xai_data

if analyze_clicked:
    if not news_text.strip():
        st.warning("Please enter a news article or headline before running the analysis.")
    else:
        with st.spinner("Running DistilBERT inference..."):
            analysis_result = run_inference(news_text.strip())
            sentiment_result = analyze_sentiment(news_text.strip())
            source_result = assess_source_credibility(st.session_state.source_name)
            explainability_items = build_reason_signals(news_text.strip(), analysis_result)
            xai_data = compute_xai_insights(news_text.strip(), analysis_result)
            append_prediction_history(analysis_result)
        st.session_state["latest_result"] = analysis_result
        st.session_state["latest_text"] = news_text.strip()
        st.session_state["latest_explainability"] = explainability_items
        st.session_state["latest_xai"] = xai_data
        st.session_state["latest_sentiment"] = sentiment_result
        st.session_state["latest_source"] = source_result


if "latest_result" in st.session_state:
    analysis_result = st.session_state["latest_result"]
    explainability_items = st.session_state["latest_explainability"]
    latest_text = st.session_state["latest_text"]
    xai_data = st.session_state.get("latest_xai", {})
    sentiment_result = st.session_state.get("latest_sentiment", {})
    source_result = st.session_state.get("latest_source", {})

    risk_label, risk_note = get_risk_profile(
        analysis_result["prediction"], analysis_result["fake_probability"]
    )
    confidence_note = get_confidence_note(analysis_result["confidence"])
    result_class = "result-real" if analysis_result["prediction"] == 1 else "result-fake"
    result_icon = "✅" if analysis_result["prediction"] == 1 else "⚠️"

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    result_left, result_right = st.columns([1.2, 1], gap="large")

    with result_left:
        st.markdown(
            """
            <div class="surface-card">
                <div class="section-title">Prediction Results</div>
                <p class="section-copy">
                    Decision, confidence, and supporting interpretation generated directly from the model output probabilities.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-banner {result_class}">
                <div class="result-title">{result_icon} {analysis_result["label"]}</div>
                <p class="result-copy">
                    {confidence_note}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_columns = st.columns(3)
        with score_columns[0]:
            st.metric("Confidence Score", f"{analysis_result['confidence']:.2f}%")
        with score_columns[1]:
            st.metric("Real Probability", f"{analysis_result['real_probability']:.2f}%")
        with score_columns[2]:
            st.metric("Fake Probability", f"{analysis_result['fake_probability']:.2f}%")

        trust_score = compute_trust_score(analysis_result)
        confidence_level = confidence_level_label(float(analysis_result["confidence"]))

        st.markdown("<div class='surface-card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="section-title">Trust Score</div>
            <p class="section-copy">
                Enterprise-style gauge derived from model probabilities (no post-processing manipulation).
            </p>
            <div style="height: 12px;"></div>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <div class="pill">{risk_label}</div>
                <div class="pill">Confidence Level: {confidence_level}</div>
            </div>
            <p class="subtle-note" style="margin-top: 10px;">{risk_note}</p>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_trust_score_gauge(trust_score, st.session_state.theme_mode),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            "<p class='subtle-note'>Trust Score uses Real Probability for REAL predictions, and 100 − Fake Probability for FAKE predictions.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="surface-card">
                <div class="section-title">Sentiment Analysis</div>
                <p class="section-copy">
                    Estimated tone classification for the submitted content (lexicon-based signal, independent of the DistilBERT prediction).
                </p>
                <div style="height: 12px;"></div>
                <div class="pill">{html.escape(sentiment_result.get("label", "Neutral"))} • {float(sentiment_result.get("confidence", 0.0)):.0f}%</div>
                <p class="subtle-note" style="margin-top: 10px;">{html.escape(sentiment_result.get("summary", ""))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        explanation_card = xai_data.get("explanation", {})
        st.markdown(
            f"""
            <div class="surface-card">
                <div class="section-title">Explainable AI</div>
                <p class="section-copy">
                    Token-level attribution is used after prediction to surface the words that most strongly supported the final class decision.
                </p>
                <div style="height: 14px;"></div>
                <div class="explanation-panel">
                    <div class="explanation-title">{html.escape(explanation_card.get("title", "Why the model predicted this class"))}</div>
                    <p class="explanation-copy">{html.escape(explanation_card.get("details", ""))}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="surface-card">
                <div class="section-title">Model Explainability</div>
                <p class="section-copy">
                    Human-readable interpretation to support portfolio presentation and reviewer understanding.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='surface-card'><ul class='explain-list'>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in explainability_items)
            + "</ul></div>",
            unsafe_allow_html=True,
        )

    with result_right:
        st.markdown(
            """
            <div class="surface-card">
                <div class="section-title">Probability Distribution</div>
                <p class="section-copy">
                    Donut visualization of class probabilities for recruiter-friendly interpretability.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_probability_chart(
                analysis_result["real_probability"],
                analysis_result["fake_probability"],
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            """
            <div class="surface-card">
                <div class="section-title">Most Influential Words</div>
                <p class="section-copy">
                    These words had the strongest attribution scores in the model explanation pass.
                </p>
                <div class="word-bank">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(build_influential_word_chips(xai_data.get("influential_words", [])), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="surface-card">
                <div class="section-title">Suspicious Words</div>
                <p class="section-copy">
                    Potentially suspicious lexical cues detected in the submitted text.
                </p>
                <div class="word-bank">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(build_suspicious_word_chips(xai_data.get("suspicious_words", [])), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        source_score = source_result.get("score")
        if source_score is None:
            score_display = "—"
        else:
            score_display = f"{int(source_score)}/100"
        st.markdown(
            f"""
            <div class="surface-card">
                <div class="section-title">News Source Credibility Score</div>
                <p class="section-copy">
                    Lightweight credibility context based on the provided publisher name. Use as supporting signal alongside fact-checking.
                </p>
                <div style="height: 12px;"></div>
                <div class="pill">{html.escape(str(source_result.get("tier", "Not provided")))} • {html.escape(score_display)}</div>
                <p class="subtle-note" style="margin-top: 10px;">{html.escape(str(source_result.get("explanation", "")))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pdf_bytes = generate_report_bytes(latest_text, analysis_result, explainability_items, xai_data)
        st.download_button(
            label="Download Analysis Report (PDF)",
            data=pdf_bytes,
            file_name="fake_news_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="surface-card">
                <div class="section-title">Professional AI Insights</div>
                <p class="section-copy">
                    The dashboard separates raw model inference from presentation logic. All displayed probabilities are sourced directly from DistilBERT softmax outputs, while the explainable AI layer adds token-level attribution without altering the classifier.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="surface-card">
            <div class="section-title">Highlighted News Review</div>
            <p class="section-copy">
                Suspicious terms are highlighted in orange, while high-impact influential words are highlighted in blue.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='highlight-box'>{xai_data.get('highlighted_text', html.escape(latest_text))}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface-card">
            <div class="section-title">Operational Note</div>
            <p class="section-copy">
                This dashboard is optimized for presentation quality and real-time inference demonstration. Final fact verification should always include trusted journalistic sources and domain validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">About Model Stack</div>
        <p class="section-copy">
            <strong>PyTorch</strong> powers efficient tensor computation during inference,
            <strong>Hugging Face Transformers</strong> provides the production-grade NLP pipeline,
            and <strong>DistilBERT</strong> delivers a compact transformer architecture well-suited for responsive text classification workloads.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">Batch News Analysis</div>
        <p class="section-copy">
            Upload a CSV of news articles to run enterprise-grade batch classification with analytics, summaries, and export.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

batch_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_csv")
batch_df = None
batch_text_cols = []

if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)
        batch_text_cols = detect_text_columns(batch_df)
        if not batch_text_cols:
            st.warning("No text columns detected in the uploaded CSV. Please include a column with article text.")
    except Exception as exc:
        st.warning(f"Unable to read CSV file: {exc}")
        batch_df = None

if batch_df is not None and batch_text_cols:
    chosen_col = st.selectbox(
        "Select the text column",
        options=batch_text_cols,
        index=0,
        help="We auto-detected text-like columns. Select the column that contains the news articles.",
    )

    run_batch_clicked = st.button("Run Batch Analysis", use_container_width=True)

    if run_batch_clicked:
        with st.spinner("Analyzing batch rows with DistilBERT..."):
            results_df, batch_summary = run_batch_analysis(batch_df, chosen_col)
        st.session_state["batch_results_df"] = results_df
        st.session_state["batch_summary"] = batch_summary

if "batch_results_df" in st.session_state:
    results_df = st.session_state["batch_results_df"]
    batch_summary = st.session_state.get("batch_summary", {})

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="surface-card">
            <div class="section-title">Batch Analytics Dashboard</div>
            <p class="section-copy">
                Summary KPIs and interactive results table for recruiter-grade product presentation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Articles", int(batch_summary.get("total", 0)))
    with kpi_cols[1]:
        st.metric("Real News Count", int(batch_summary.get("real_count", 0)))
    with kpi_cols[2]:
        st.metric("Fake News Count", int(batch_summary.get("fake_count", 0)))
    with kpi_cols[3]:
        st.metric("Average Confidence", f"{float(batch_summary.get('avg_confidence', 0.0)):.2f}%")

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_bytes,
        file_name=f"veritas_ai_batch_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="surface-card">
        <div class="section-title">Prediction History</div>
        <p class="section-copy">
            Recent analyses stored in Streamlit session state (timestamp, prediction, and confidence).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

history_rows = list(reversed(st.session_state.prediction_history))
if history_rows:
    st.dataframe(history_rows, use_container_width=True, hide_index=True)
else:
    st.caption("No analyses recorded yet. Run an analysis to populate the history table.")
