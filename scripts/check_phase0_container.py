"""Run Phase 0 checks inside the disposable Python 3.11 container."""

import subprocess
import sys


def main() -> int:
    """Install test tooling and stop immediately if any check fails."""
    if sys.version_info[:2] != (3, 11):
        sys.stderr.write("Run this helper in the project's Python 3.11 Docker image.\n")
        return 1

    commands = [
        [
            sys.executable, "-m", "pip", "install",
            "pytest>=8,<10", "httpx>=0.27,<1", "ruff>=0.8,<1", "pytest-timeout>=2,<3",
        ],
        [sys.executable, "--version"],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "pytest", "-q", "--timeout=60"],
    ]
    for command in commands:
        # Argument lists preserve version constraints without shell interpretation.
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
