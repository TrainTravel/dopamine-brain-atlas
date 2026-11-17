# Builds the site and runs the whole test suite. NOT yet run by its author: verify before relying on it.
#   docker build -t atlas . && docker run --rm atlas            # run tests
#   docker run --rm -p 8080:8080 atlas python3 -m http.server 8080 -d dist   # preview the site
FROM mcr.microsoft.com/playwright:v1.63.0-noble
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt package.json package-lock.json ./
RUN pip3 install --break-system-packages -r requirements.txt && npm ci
COPY . .
RUN ./scripts/fetch_neurosynth.sh && ./scripts/fetch_model.sh
CMD ["./run_tests.sh"]
