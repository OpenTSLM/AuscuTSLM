#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Check OpenAI API credit balance and usage.

Usage:
    python scripts/utils/check_openai_balance.py

    # Or with explicit API key
    python scripts/utils/check_openai_balance.py --api_key sk-proj-your-key
"""

import argparse
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from openai import OpenAI
import requests


def _query_billing(api_key):
    """
    Try multiple OpenAI billing endpoints to retrieve credit / usage info.
    Returns a dict with whatever we could fetch, or empty dict on failure.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    info = {}

    # --- Endpoint 1: /dashboard/billing/credit_grants (legacy, may still work) ---
    try:
        r = requests.get(
            "https://api.openai.com/dashboard/billing/credit_grants",
            headers=headers, timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            info["total_granted"] = data.get("total_granted", 0)
            info["total_used"] = data.get("total_used", 0)
            info["total_available"] = data.get("total_available",
                                                info["total_granted"] - info["total_used"])
    except Exception:
        pass

    # --- Endpoint 2: /dashboard/billing/subscription ---
    try:
        r = requests.get(
            "https://api.openai.com/dashboard/billing/subscription",
            headers=headers, timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            info["plan"] = data.get("plan", {}).get("title", "unknown")
            limit = data.get("hard_limit_usd")
            if limit is not None:
                info["hard_limit_usd"] = limit
    except Exception:
        pass

    # --- Endpoint 3: /dashboard/billing/usage (last 30 days) ---
    try:
        today = datetime.now(timezone.utc)
        start = today - timedelta(days=30)
        params = {
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": today.strftime("%Y-%m-%d"),
        }
        r = requests.get(
            "https://api.openai.com/dashboard/billing/usage",
            headers=headers, params=params, timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            total_cents = data.get("total_usage", 0)
            info["usage_last_30d"] = total_cents / 100.0  # convert cents → dollars
    except Exception:
        pass

    return info


def check_balance(api_key=None):
    """
    Check OpenAI API credit balance and print it.

    Args:
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
    """
    # Get API key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ No API key provided!")
        print("\nPlease either:")
        print("  1. Set environment variable: export OPENAI_API_KEY='your-key'")
        print("  2. Pass as argument: --api_key your-key")
        sys.exit(1)

    # Mask key for display
    masked_key = api_key[:10] + "..." + api_key[-4:]
    print(f"Using API key: {masked_key}\n")

    # Test API key with a lightweight request (list models — no tokens used)
    print("Testing API key...")
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        print("✅ API key is valid!\n")
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate_limit" in error_str.lower():
            print("✅ API key is valid! (rate-limited right now, but that's OK)\n")
        elif "401" in error_str or "invalid" in error_str.lower():
            print(f"❌ API key is invalid: {e}\n")
            sys.exit(1)
        else:
            # Other errors (network, etc.) — warn but continue
            print(f"⚠️  Could not verify key ({type(e).__name__}), continuing anyway...\n")

    # ── Query billing information ──
    print("=" * 60)
    print("💳 OPENAI CREDIT BALANCE")
    print("=" * 60)

    billing = _query_billing(api_key)

    if billing.get("total_available") is not None:
        print(f"\n  Total granted:   ${billing['total_granted']:.2f}")
        print(f"  Total used:      ${billing['total_used']:.2f}")
        print(f"  ✅ Remaining:     ${billing['total_available']:.2f}")
    elif billing.get("hard_limit_usd") is not None:
        print(f"\n  Monthly limit:   ${billing['hard_limit_usd']:.2f}")
        if billing.get("usage_last_30d") is not None:
            remaining = billing["hard_limit_usd"] - billing["usage_last_30d"]
            print(f"  Used (30 days):  ${billing['usage_last_30d']:.2f}")
            print(f"  ✅ Remaining:     ${remaining:.2f}")
    elif billing.get("usage_last_30d") is not None:
        print(f"\n  Usage (last 30 days): ${billing['usage_last_30d']:.2f}")
    else:
        print("\n  ⚠️  Could not retrieve balance programmatically.")
        print("     Check manually: https://platform.openai.com/usage")
        print("\n  💡 Actual token usage & cost will be printed after generation completes.")

    if billing.get("plan"):
        print(f"  Plan: {billing['plan']}")

    # ── Cost estimates ──
    print("\n" + "=" * 60)
    print("COST ESTIMATE FOR AUDIO COT GENERATION")
    print("=" * 60)

    costs = [
        ("10 samples (test)", 10, 0.0004),
        ("100 samples", 100, 0.04),
        ("1,000 samples", 1000, 0.40),
        ("10,000 samples", 10000, 4.00),
        ("26,100 samples (train)", 26100, 10.44),
        ("32,620 samples (full)", 32620, 13.05),
    ]

    print("\nUsing GPT-4o-mini + Spectrograms:")
    print("-" * 60)
    for desc, count, cost in costs:
        print(f"  {desc:30s}  ${cost:6.2f}")

    print("\n" + "=" * 60)
    print("QUICK PRICING REFERENCE")
    print("=" * 60)
    print("\nGPT-4o-mini:")
    print("  Input:  $0.150 / 1M tokens")
    print("  Output: $0.600 / 1M tokens")
    print("  Images: $0.255 / 1M tokens (fixed ~765 tokens per image)")

    print("\nGPT-4o:")
    print("  Input:  $2.50 / 1M tokens")
    print("  Output: $10.00 / 1M tokens")
    print("  Images: $2.55 / 1M tokens")

    print("\n" + "=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check OpenAI API balance and estimate costs"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()
    check_balance(api_key=args.api_key)


if __name__ == "__main__":
    main()
