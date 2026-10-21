"""User-facing error types and translation of OpenAI SDK errors into plain language."""
from __future__ import annotations


class UserFacingError(Exception):
    """An error whose message is safe and useful to show in the UI."""


def explain_openai_error(exc: Exception, stage: str) -> UserFacingError:
    """Map OpenAI SDK exceptions to actionable messages. Never leaks the API key."""
    import openai

    prefix = f"{stage} failed: "
    body = getattr(exc, "body", None) or {}
    if isinstance(body, dict) and "error" in body and isinstance(body["error"], dict):
        body = body["error"]
    code = (body.get("code") if isinstance(body, dict) else None) or getattr(exc, "code", None)
    api_msg = body.get("message") if isinstance(body, dict) else None

    if isinstance(exc, openai.RateLimitError) and code == "insufficient_quota":
        return UserFacingError(
            prefix + "the OpenAI account has run out of credit (insufficient_quota). "
            "Add credit or raise the usage limit at platform.openai.com → Billing, then retry. "
            "Cached steps will not be re-billed."
        )
    if isinstance(exc, openai.RateLimitError):
        return UserFacingError(prefix + "OpenAI rate limit reached even after retries. Wait a minute and try again.")
    if isinstance(exc, openai.AuthenticationError):
        return UserFacingError(
            prefix + "the OpenAI API key is missing or invalid. Set the OPENAI_API_KEY secret in the Space settings."
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return UserFacingError(prefix + "this OpenAI key/project has no access to the configured model.")
    if isinstance(exc, openai.NotFoundError):
        return UserFacingError(prefix + "the configured OpenAI model was not found. Check TRANSCRIBE_MODEL / LLM_MODEL.")
    if isinstance(exc, openai.APITimeoutError):
        return UserFacingError(prefix + "OpenAI did not respond in time. Please retry.")
    if isinstance(exc, openai.APIConnectionError):
        return UserFacingError(prefix + "could not reach OpenAI (network problem). Please retry.")
    if isinstance(exc, openai.BadRequestError):
        return UserFacingError(prefix + f"OpenAI rejected the request: {api_msg or exc}")
    if isinstance(exc, openai.InternalServerError):
        return UserFacingError(prefix + "OpenAI had a server error. Please retry shortly.")
    return UserFacingError(prefix + str(exc))
