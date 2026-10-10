# 0001 — Replay historical data as live

**Status:** Accepted (2026-10-10)

> **Update (2026-10-10, trim, [0009](0009-trim-for-defensibility.md)):** the OpenAQ client was removed (it was never used without a key). Replay is now the only source; adding a live source means writing a new client that produces the same rows.

## Context
OpenAQ v3 needs a free API key (header `X-API-Key`), which only Juthy can register for. Without live data the poller, read API, assistant and edge simulator would have nothing to work on.

## Decision
The poller has two sources. `openaq` calls `/v3/locations?coordinates=…&radius=25000` and `/v3/locations/{id}/latest`. `replay` serves the Mendeley Bangladesh AQI dataset (doi:10.17632/9j447cynb9.1, CC BY 4.0) as live: the reading for hour H is the dataset's value at H − 730 days, re-stamped to H and stored with `replayed_from`. Every replayed reading carries `source: replay-mendeley`, which the API and the assistant always show.

## Finding that shaped this
The dataset's column names match Open-Meteo's air-quality API (CAMS model output), and Dhaka's series has no missing hour from 2000 to 2025. These are almost certainly **modelled** values, not sensor readings (an inference — the publisher does not say). They are fine for exercising the system and for injecting faults with exact labels; they are never presented as measurements. The dataset is not truncated (3.19 M rows, not the 1,048,551 the research worried about).

## Consequences
- ✅ The whole system runs before any account exists; switching to live is one parameter.
- ⚠️ Anomaly results are on smooth modelled data plus synthetic faults; real low-cost sensors are noisier.
