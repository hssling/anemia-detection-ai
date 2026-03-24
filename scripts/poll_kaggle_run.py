# scripts/poll_kaggle_run.py
"""
Poll a Kaggle kernel run until it completes or fails.
Exits with code 0 on success, 1 on failure/timeout.

Usage:
    python scripts/poll_kaggle_run.py \
        --kernel <username>/<kernel_slug> \
        --timeout-minutes 360
"""
import argparse
import logging
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApiExtended

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 60
TERMINAL_STATUSES = {"complete", "error", "cancel"}


def poll(kernel_ref: str, timeout_minutes: int) -> bool:
    """Poll kernel status. Returns True on success, False on failure/timeout."""
    api = KaggleApiExtended()
    api.authenticate()

    deadline = time.time() + timeout_minutes * 60
    log.info(f"Polling kernel: {kernel_ref} (timeout: {timeout_minutes} min)")

    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    while time.time() < deadline:
        try:
            status_obj = api.kernels_status(kernel_ref)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            log.warning(f"  API error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {exc}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error("✗ Too many consecutive API errors — aborting poll.")
                return False
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        status = status_obj.get("status", "unknown").lower()
        log.info(f"  Status: {status}")

        if status == "complete":
            log.info("✓ Kernel completed successfully.")
            return True
        elif status in {"error", "cancel"}:
            log.error(f"✗ Kernel ended with status: {status}")
            return False

        time.sleep(POLL_INTERVAL_SECONDS)

    log.error(f"✗ Timeout after {timeout_minutes} minutes.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Poll Kaggle kernel run")
    parser.add_argument("--kernel", required=True, help="username/kernel_slug")
    parser.add_argument("--timeout-minutes", type=int, default=360)
    args = parser.parse_args()

    success = poll(args.kernel, args.timeout_minutes)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
