"""Per-category advice parsed from the AirNow "Air Quality Guide for Particle
Pollution" table (page 1).

Vector search over PDF table text is unreliable (rows split across chunks),
and a 1.5B model given the whole table often answers for the wrong row. So
the table is parsed once at ingest into {category: advice} and the assistant
is handed exactly the row for the reading's category as source [1].
"""

import re

CATEGORIES = [
    ("Good", "(0-50)"),
    ("Moderate", "(51-100)"),
    ("Unhealthy for Sensitive Groups", "(101-150)"),
    ("Unhealthy", "(151-200)"),
    ("Very Unhealthy", "(201-300)"),
    ("Hazardous", "(301-500)"),
]


def parse(page_text: str) -> dict[str, str]:
    flat = re.sub(r"\s+", " ", page_text)
    flat = flat.replace("It ’s", "It’s")
    starts = []
    for name, rng in CATEGORIES:
        m = re.search(re.escape(name) + r"\s*" + re.escape(rng), flat)
        if m:
            starts.append((m.start(), m.end(), name))
    starts.sort()
    out = {}
    for i, (_start, e, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else flat.find("*Note")
        out[name] = flat[e:end].strip()
    return out
