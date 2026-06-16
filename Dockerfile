FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY config ./config
COPY src ./src

RUN python -m pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "ai_trading.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
