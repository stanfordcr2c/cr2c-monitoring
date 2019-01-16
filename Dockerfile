FROM python:alpine3.7
COPY . /app
WORKDIR /app
RUN apk add make automake gcc g++ subversion python3-dev
RUN pip install -r requirements.txt


# Test Locally
CMD ["gunicorn", "-b","0.0.0.0:8080", "main:server", "-t", "3600"]

# Deploy to GAE
# CMD exec gunicorn -b :$PORT main:server --timeout 1800
