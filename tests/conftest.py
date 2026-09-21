import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# tests always run on the mock, whatever .env says
for k in list(os.environ):
    if k.startswith("LLM_"):
        del os.environ[k]
