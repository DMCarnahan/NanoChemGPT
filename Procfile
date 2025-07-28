# procfile for railway interface

web: gunicorn -k gthread --threads 8 --workers 1 --timeout 180 --graceful-timeout 30 \
  -b 0.0.0.0:$PORT --access-logfile - --error-logfile - app:app
