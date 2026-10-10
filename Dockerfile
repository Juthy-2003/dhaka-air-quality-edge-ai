# The app image: FastAPI app (readings, anomalies, /ask, model manifest) and
# the SYNTHETIC sensor simulator. Training libraries (scikit-learn) are a
# separate dependency group and are not installed here.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/venv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups

FROM python:3.12-slim
RUN useradd --uid 1000 --create-home app
COPY --from=build /venv /venv
COPY --from=build /app/src /app/src
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 AQEDGE_HOME=/app AQEDGE_DATA=/data HF_HOME=/tmp/hf \
    PUBLISHED_MODELS=/app/models/published
WORKDIR /app
USER app
EXPOSE 8000
CMD ["uvicorn", "aqedge.app:app", "--host", "0.0.0.0", "--port", "8000"]
