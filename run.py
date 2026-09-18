"""Convenience entrypoint for local development: `python run.py`.

In production, prefer running uvicorn/gunicorn directly (see Dockerfile),
so that worker count, reload, and TLS termination are controlled explicitly.
"""
import webbrowser

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    url = f"http://localhost:{settings.port}"
    print(f"\n{'=' * 70}\n {settings.app_name} v{settings.version}\n{'=' * 70}")
    print(f"[+] Serving on {url} (environment={settings.environment})\n")

    if settings.environment == "development":
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )
