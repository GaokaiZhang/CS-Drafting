import math
import re


_CHOICE_RE_LIST = [
    re.compile(r"answer\s*(?:is|:)\s*\(?([ABCD])\)?", re.IGNORECASE),
    re.compile(r"option\s*\(?([ABCD])\)?", re.IGNORECASE),
    re.compile(r"\(([ABCD])\)"),
]
_STRICT_CHOICE_RE_LIST = _CHOICE_RE_LIST[:]
_FINAL_NUMBER_RE_LIST = [
    re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"answer\s*(?:is|:)\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
]

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(text):
    value = text.replace(",", "").strip()
    if not value:
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    if math.isfinite(num) and num.is_integer():
        return str(int(num))
    return str(num)


def extract_mmlu_choice(text):
    if not text:
        return None
    tail = text[-400:]
    for pattern in _CHOICE_RE_LIST:
        matches = pattern.findall(tail)
        if matches:
            return matches[-1].upper()
    letters = re.findall(r"\b([ABCD])\b", tail)
    if letters:
        return letters[-1].upper()
    return None


def extract_mmlu_choice_strict(text):
    if not text:
        return None
    tail = text[-400:]
    for pattern in _STRICT_CHOICE_RE_LIST:
        matches = pattern.findall(tail)
        if matches:
            return matches[-1].upper()
    return None


def extract_gsm8k_answer(text):
    if not text:
        return None
    if "####" in text:
        maybe = text.rsplit("####", 1)[-1]
        match = _NUMBER_RE.search(maybe)
        if match:
            return _normalize_number(match.group(0))
    matches = _NUMBER_RE.findall(text)
    if matches:
        return _normalize_number(matches[-1])
    return None


def extract_gsm8k_answer_strict(text):
    if not text:
        return None
    tail = text[-400:]
    for pattern in _FINAL_NUMBER_RE_LIST:
        matches = pattern.findall(tail)
        if matches:
            return _normalize_number(matches[-1])
    return None


def score_mmlu(item, generated_text):
    predicted_choice = extract_mmlu_choice(generated_text)
    gold_choice = "ABCD"[int(item["answer"])]
    correct = predicted_choice == gold_choice
    return {
        "metric_name": "accuracy",
        "prediction": predicted_choice,
        "gold": gold_choice,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
    }


def score_gsm8k(item, generated_text):
    predicted_answer = extract_gsm8k_answer(generated_text)
    gold_text = item["answer"]
    gold_answer = extract_gsm8k_answer(gold_text)
    correct = predicted_answer is not None and predicted_answer == gold_answer
    return {
        "metric_name": "exact_match",
        "prediction": predicted_answer,
        "gold": gold_answer,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
    }


def score_sample(dataset_name, item, generated_text):
    if dataset_name == "mmlu":
        return score_mmlu(item, generated_text)
    if dataset_name == "gsm8k":
        return score_gsm8k(item, generated_text)
    raise ValueError(f"Unsupported dataset for scoring: {dataset_name}")


def has_final_answer_signal(dataset_name, generated_text):
    if dataset_name == "mmlu":
        return extract_mmlu_choice_strict(generated_text) is not None
    if dataset_name == "gsm8k":
        return extract_gsm8k_answer_strict(generated_text) is not None
    return False
