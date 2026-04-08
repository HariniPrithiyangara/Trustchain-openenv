FROM python:3.11-slim

# Set up user 1000 for Hugging Face compatibility
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Set environment variables
ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true
EXPOSE 7860

# Run the application
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
