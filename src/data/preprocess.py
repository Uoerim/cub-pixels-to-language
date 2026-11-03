import html
import re
import unicodedata


def preprocess_captions(description):
    if not isinstance(description, str):
        return []
    captions = []
    for line in description.splitlines():
        text = unicodedata.normalize("NFKC", html.unescape(line)).lower()
        text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
        text = " ".join(text.split())
        if text:
            captions.append(text)
    return captions


def caption_anomalies(description, expected_count=10):
    captions = preprocess_captions(description)
    raw = description if isinstance(description, str) else ""
    issues = []
    if not isinstance(description, str):
        issues.append("non_string_description")
    if len(captions) != expected_count:
        issues.append("unexpected_caption_count")
    if len(set(captions)) != len(captions):
        issues.append("repeated_caption")
    if any(char.isdigit() for char in raw):
        issues.append("contains_digits")
    if "\ufffd" in raw or any(ord(char) < 32 and char not in "\r\n\t" for char in raw):
        issues.append("control_or_replacement_character")
    if any(len(text.split()) < 3 for text in captions):
        issues.append("short_caption")
    return captions, issues
