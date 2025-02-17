#!/bin/bash

# Number of workers
WORKERS=4

# Host and port
HOST=0.0.0.0
PORT=9001

# Run Gunicorn with Uvicorn worker
gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker api.search_context_api:app --bind $HOST:$PORT