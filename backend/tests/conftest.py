import os

os.environ.setdefault("ENABLE_MODEL_WARMUP", "false")
os.environ.setdefault("ENABLE_SQLITE_FALLBACK", "true")
os.environ.setdefault("SQLITE_PATH", "./backend/data/test.db")
os.environ.setdefault("RATE_LIMIT_ANALYZE", "1000/minute")
os.environ.setdefault("RATE_LIMIT_SPEECH", "1000/minute")
