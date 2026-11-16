FROM python:3.14-slim

# uv binary directory
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /code

COPY pyproject.toml uv.lock ./

# production deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev \
    --allow-insecure-host pypi.org \
    --allow-insecure-host files.pythonhosted.org

ENV PATH="/code/.venv/bin:$PATH"

COPY ./app ./app

# Create and switch to a non-root user
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /code
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]