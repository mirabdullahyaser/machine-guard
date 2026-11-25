"""API key check for /predict.

The key lives in the API_KEY environment variable, never in code or Git. Callers send it in the
X-API-Key header. A wrong or missing key gets 401 Unauthorized.
"""
import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# auto_error=False: we raise 401 ourselves (FastAPI's default would be 403 for a missing header).
# Using APIKeyHeader also gives /docs its "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(api_key_header)) -> None:
    expected = os.environ.get("API_KEY")  # read on every request, so tests can set it late
    if not expected:
        # Fail closed: a server without a key configured must not serve predictions to anyone.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Server has no API_KEY configured")
    # compare_digest takes the same time whether the first or the last character is wrong,
    # so an attacker cannot guess the key one character at a time from response times.
    # .encode() compares bytes: compare_digest raises on non-ASCII text, which would be a 500.
    if not api_key or not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key",
                            headers={"WWW-Authenticate": "API-Key"})
