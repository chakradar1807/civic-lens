"""Multilingual complaint intake (English, Tamil, Hindi - script and romanised).

A transparent keyword baseline: category + urgency + language guess + matched keywords.
It is intentionally simple and explainable; replace with a fine-tuned multilingual model
(e.g. IndicBERT / MuRIL) when labelled complaints are available.
"""
import re
from typing import Dict, List

KEYWORDS = {
    "pothole": [
        "pothole", "potholes", "road damage", "crater", "broken road", "hole in road", "hole in the road",
        "kuzhi", "gundu kuzhi", "road la kuzhi", "sadagu", "gaddha", "gadda", "sadak toot",
        "குழி", "பள்ளம்", "சாலை", "गड्ढा", "गड्ढे", "सड़क",
    ],
    "garbage": [
        "garbage", "trash", "waste", "litter", "dump", "dustbin", "overflowing bin", "smell",
        "kuppai", "kuppa", "kuppathotti", "kachra", "kooda",
        "குப்பை", "குப்பைத்தொட்டி", "कचरा", "कूड़ा", "गंदगी",
    ],
    "damaged_streetlight": [
        "streetlight", "street light", "lamp", "light not working", "dark street", "no light", "bulb",
        "vilakku", "vilakku erila", "batti", "light kharab", "andhera",
        "விளக்கு", "தெரு விளக்கு", "இருட்டு", "बत्ती", "लाइट", "स्ट्रीट लाइट", "अंधेरा",
    ],
    "waterlogging": [
        "waterlogging", "water logging", "flood", "flooded", "water stagnation", "drain blocked",
        "sewage", "overflow", "drainage",
        "thanni", "thanneer", "mazhai neer", "vellam", "pani bhar", "naali", "jal jamav", "paani",
        "தண்ணீர்", "வெள்ளம்", "மழைநீர்", "கழிவுநீர்", "पानी", "जलभराव", "बाढ़", "नाली",
    ],
}

URGENT_WORDS = [
    "urgent", "emergency", "danger", "dangerous", "accident", "injured", "hospital", "school",
    "children", "fatal", "electric shock", "live wire", "exposed wire", "immediately", "asap",
    "aapatkal", "abathu", "avasaram", "khatra", "turant",
    "ஆபத்து", "அவசரம்", "விபத்து", "खतरा", "तुरंत", "दुर्घटना", "अस्पताल",
]


def _detect_language(text: str) -> str:
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en/romanised"


def _count_hits(text: str, words: List[str]) -> List[str]:
    hits = []
    for w in words:
        if w.isascii():
            if re.search(r"\b" + re.escape(w) + r"\b", text):
                hits.append(w)
        elif w in text:
            hits.append(w)
    return hits


def classify_complaint(text: str) -> Dict:
    clean = (text or "").strip()
    lowered = clean.lower()
    scores = {cat: _count_hits(lowered, words) for cat, words in KEYWORDS.items()}
    best_cat, best_hits = max(scores.items(), key=lambda kv: len(kv[1]))
    total_hits = sum(len(v) for v in scores.values())

    urgent_hits = _count_hits(lowered, URGENT_WORDS)
    urgency = "high" if len(urgent_hits) >= 2 else "medium" if urgent_hits else "low"

    if not best_hits:
        category, confidence = "unknown", 0.0
    else:
        category = best_cat
        confidence = round(len(best_hits) / total_hits, 2)

    return {
        "category": category,
        "confidence": confidence,
        "urgency": urgency,
        "language": _detect_language(clean),
        "matched_keywords": best_hits,
        "urgent_keywords": urgent_hits,
        "needs_human_review": category == "unknown" or confidence < 0.6,
        "method": "keyword baseline (explainable); replace with fine-tuned multilingual model",
    }
