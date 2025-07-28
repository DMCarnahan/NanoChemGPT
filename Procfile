web: gunicorn -k gthread --threads 8 --workers 1 --timeout 600 --graceful-timeout 60 -b 0.0.0.0:$PORT --access-logfile - --error-logfile - app:app
