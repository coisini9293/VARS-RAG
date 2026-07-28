"""Parse model text into Offence / No offence / Between / Unknown."""

import re
from collections import Counter

from experiment.constants import LABELS


def parse_offence_label(text: str) -> str:
    lines = text.strip().splitlines()
    blob = (lines[0] if lines else text).lower()
    full = text.lower()
    for scope in (blob, full):
        if "no offence" in scope or "no offense" in scope:
            return "No offence"
        if re.search(r"\bbetween\b", scope):
            return "Between"
        if "offence" in scope or "offense" in scope or "foul" in scope:
            if "no offence" not in scope and "no offense" not in scope:
                return "Offence"
    m = re.search(r"final:\s*(offence|no offence|no offense|between)", text.lower())
    if m:
        s = m.group(1)
        if "no" in s:
            return "No offence"
        if "between" in s:
            return "Between"
        return "Offence"
    return "Unknown"


def majority_label(votes: list[str]) -> str:
    valid = [x for x in votes if x in LABELS]
    if not valid:
        return "Unknown"
    return Counter(valid).most_common(1)[0][0]
