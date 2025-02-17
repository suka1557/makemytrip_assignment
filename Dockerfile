FROM python:3.10

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 9001

# Run the startup script
CMD ["/bin/bash", "run_api.sh"]
