FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install . && useradd --create-home --uid 10001 app && chown -R app /app
USER app

# Generate data + train the small model + tune the threshold at build time so the image is
# self-contained (~30 s). Mount ./artifacts over /app/artifacts to serve your own model instead.
RUN modelroute train && modelroute eval --docs-dir /tmp/docs

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["modelroute", "--log-level", "INFO", "serve", "--port", "8000"]
