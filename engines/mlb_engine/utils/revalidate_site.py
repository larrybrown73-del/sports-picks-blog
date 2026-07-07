#!/usr/bin/env python3
"""POST to the Next.js ISR revalidation endpoint for key site paths."""

from __future__ import annotations

import os
import sys

import requests

DEFAULT_PATHS = ["/", "/picks", "/performance"]


def revalidate_site(paths: list[str] | None = None) -> int:
    secret = os.environ.get("REVALIDATION_SECRET", "").strip()
    site_url = os.environ.get("SITE_URL", "").strip().rstrip("/")
    if not secret or not site_url:
        print("Skip revalidation: REVALIDATION_SECRET or SITE_URL not set.")
        return 0

    for page_path in paths or DEFAULT_PATHS:
        response = requests.post(
            f"{site_url}/api/revalidate",
            params={"secret": secret, "path": page_path},
            timeout=30,
        )
        print(f"Revalidate {page_path}: {response.status_code} {response.text[:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(revalidate_site())
