"""US EPA AQI category for PM2.5 — deterministic, so the assistant never has
to do arithmetic. Breakpoints from the EPA AQI Technical Assistance Document,
EPA-403/B-26-003 (May 2026), Table 6, which is also in the corpus.

Caveat stated in every answer: EPA categories use 24-hour averages (or the
NowCast for hourly reporting); applying them to one hourly value is an
approximation.
"""

PM25_BREAKPOINTS = [  # (low, high, category)
    (0.0, 9.0, "Good"),
    (9.1, 35.4, "Moderate"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups"),
    (55.5, 125.4, "Unhealthy"),
    (125.5, 225.4, "Very Unhealthy"),
    (225.5, float("inf"), "Hazardous"),
]


def pm25_category(value: float) -> str:
    v = round(value, 1)
    for low, high, name in PM25_BREAKPOINTS:
        if low <= v <= high:
            return name
    return "Good" if v < 0 else "Hazardous"
