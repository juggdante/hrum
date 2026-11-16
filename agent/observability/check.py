"""Check the Langfuse connection and send one test trace.

    python -m agent.observability.check
"""
from __future__ import annotations

import sys

from agent.observability import tracing


def main() -> None:
    host = tracing.base_url() or "(SDK default)"
    if not tracing.enabled():
        sys.exit("Tracing is off: set LANGFUSE_HOST (or LANGFUSE_BASE_URL), LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env "
                 "(see infra/langfuse/README.md).")
    client = tracing.client()
    try:
        ok = client.auth_check()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Cannot reach Langfuse at {host}: {type(e).__name__}: {e}")
    if not ok:
        sys.exit(f"Langfuse at {host} rejected the keys (check public/secret key and project).")
    with tracing.trace_attributes(trace_name="hrum.check", tags=["check"]), \
         tracing.observation("check", as_type="span", input="Berapa gaji Stephen Lee? (masking test)") as obs:
        obs.update(output="ok")
        url = tracing.trace_url()
    tracing.flush()
    print(f"Langfuse OK at {host}. Test trace sent (the name in its input should show as [NAME]).")
    if url:
        print(f"Open: {url}")


if __name__ == "__main__":
    main()
