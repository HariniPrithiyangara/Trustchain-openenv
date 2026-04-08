FROM python:3.10-slim

WORKDIR /app

# Install project
COPY pyproject.toml .
COPY server/requirements.txt* ./server/
RUN pip install --no-cache-dir -e .

# Copy environment code
COPY server/ ./server/
COPY openenv.yaml .

# OpenEnv spec
ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "server.app"]
