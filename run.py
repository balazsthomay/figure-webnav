"""Entry point: uv run run.py"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src/ is on the path for direct execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv


async def main(headless: bool = True) -> int:
    load_dotenv()

    from webnav.agent import Agent

    agent = Agent(headless=headless)
    metrics = await agent.run()

    # Exit code: 0 if >=28/30 solved, 1 otherwise
    return 0 if metrics.steps_succeeded >= 28 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Navigation Challenge Agent")
    parser.add_argument("--no-headless", action="store_true", help="Run with visible browser")
    args = parser.parse_args()

    exit_code = asyncio.run(main(headless=not args.no_headless))
    sys.exit(exit_code)
