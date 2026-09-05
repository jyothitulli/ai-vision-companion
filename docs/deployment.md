# Deployment

1. Copy `.env.example` to `.env` and set a strong `API_KEY`.
2. `docker compose up --build` starts PostgreSQL + API.
3. Point the Expo app at the API (`EXPO_PUBLIC_API_URL`).
4. Keep CORS origins explicit. Do not use `*` in production.
5. Place model weights in a volume if you want to avoid re-downloading.
6. Reverse-proxy TLS in front of port 8000.

CPU-only containers are expected until a CUDA image is provided. Measure latency before advertising real-time performance.
