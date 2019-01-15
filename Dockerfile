FROM python:alpine3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt


# Test Locally
CMD ["gunicorn", "-b","0.0.0.0:8080", "main:server", "-t", "3600"]

# Deploy to GAE