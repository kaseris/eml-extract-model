import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .errors import LLMRateLimitError


def make_retry_on_rate_limit(component: str, logger: logging.Logger):
    """Return a tenacity retry decorator that retries only on LLMRateLimitError.

    Config: 3 attempts total, exponential backoff starting at 1 s, capped at 8 s.
    After all attempts are exhausted LLMRateLimitError is re-raised unchanged
    (reraise=True — never swallowed or wrapped).
    """
    def _before_sleep(retry_state):
        logger.warning(
            '%s: rate limit hit, retrying (attempt %d of 3) after %.1fs',
            component,
            retry_state.attempt_number,
            retry_state.next_action.sleep,
        )

    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(LLMRateLimitError),
        reraise=True,
        before_sleep=_before_sleep,
    )
