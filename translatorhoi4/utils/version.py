"""Version info.

Version is determined in the following order:
1. From environment variable APP_VERSION (set by GitHub Actions during release builds)
2. Fallback to 'dev' for local/development builds
"""
from __future__ import annotations

import os


def get_version() -> str:
    """Get the application version.

    Returns:
        Version string (e.g., '1.7' for releases, 'dev' for local builds)
    """
    # 1. Check environment variable (set by CI/CD)
    env_version = os.environ.get('APP_VERSION')
    if env_version:
        return env_version

    # 2. Fallback to dev version for local builds
    return 'dev'


# Module-level variable for easy access
__version__ = get_version()
