"""Find which monitored location a question is about.

The readings table has four Dhaka-area locations. People ask about
neighbourhoods, so a small gazetteer maps common neighbourhood names to
coordinates, and the nearest monitored location answers for them. The answer
always says which location's reading was used.
"""

import math
import re

MONITORED = {  # name → (lat, lon); from the replay dataset's cities.csv
    "Dhaka": (23.7104, 90.40744),
    "Azimpur": (23.7298, 90.3854),
    "Paltan": (23.7362, 90.4137),
    "Tungi": (23.8915, 90.4023),
}
NEIGHBOURHOODS = {  # approximate centres (OpenStreetMap, rounded)
    "mirpur": (23.8042, 90.3667),
    "gulshan": (23.7925, 90.4078),
    "banani": (23.7937, 90.4066),
    "uttara": (23.8759, 90.3795),
    "dhanmondi": (23.7465, 90.3760),
    "mohammadpur": (23.7662, 90.3589),
    "motijheel": (23.7330, 90.4172),
    "tejgaon": (23.7639, 90.3900),
    "badda": (23.7805, 90.4267),
    "old dhaka": (23.7104, 90.4074),
    "jatrabari": (23.7104, 90.4349),
    "tongi": (23.8915, 90.4023),
    "paltan": (23.7362, 90.4137),
    "azimpur": (23.7298, 90.3854),
    "lalbagh": (23.7189, 90.3882),
}


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * math.cos(math.radians(a[0])))


def resolve(question: str) -> tuple[str, str] | None:
    """Return (place mentioned, monitored location used), or None."""
    q = question.lower()
    for name in MONITORED:
        if re.search(rf"\b{name.lower()}\b", q) and name != "Dhaka":
            return name, name
    for hood, coords in NEIGHBOURHOODS.items():
        if re.search(rf"\b{hood}\b", q):
            nearest = min(MONITORED, key=lambda m: _dist(coords, MONITORED[m]))
            return hood.title(), nearest
    if re.search(r"\bdhaka\b", q):
        return "Dhaka", "Dhaka"
    return None
