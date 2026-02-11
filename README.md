# figure-webnav

Agent that solves the [30-step browser navigation challenge](https://serene-frangipane-7fd25b.netlify.app/) in under 5 minutes.

## Quick Start

```bash
# Prerequisites: Python 3.12+, uv

# Install dependencies
uv sync

# Install browser
uv run playwright install chromium

# Set up API key (OpenRouter)
echo "OPENROUTER_API_KEY=your-key-here" > .env

# Run (headless)
uv run python run.py

# Run (visible browser)
uv run python run.py --no-headless
```

## Run Statistics (sample)

```
Completed: 30/30
Total time: 229.3s
LLM calls: 4
LLM tokens: 7476
Est. cost: $0.0022
Avg time/step: 6.0s
```

Per-step breakdown with timing, tier used, and codes is printed after each run.

## Architecture

Three-tier dispatch minimizes LLM usage — most steps resolve in <2s with zero API calls.

| Tier | Method | Latency | When Used |
|------|--------|---------|-----------|
| 0 | Regex on instruction text | ~0ms | 90% of steps |
| 1 | Gemini 3 Flash (OpenRouter) | ~0.5-1s | Novel challenge types |
| 2 | Claude Haiku 4.5 + screenshot | ~2-3s | Stuck recovery (3+ failures) |

### Pipeline per step

1. **Clean** — JS injection removes popup overlays (z-index > 500)
2. **Perceive** — Extract instruction + codes from page text and aria snapshot
3. **Dispatch** — Tier 0 regex match, fallback to Tier 1/2 LLM
4. **Execute** — Playwright actions (click, scroll, hover, drag, canvas draw, key sequence, JS eval)
5. **Extract** — Scan for 6-char alphanumeric code in DOM text, attributes, comments, green success boxes
6. **Submit** — Fill code input + click Submit
7. **Verify** — URL change confirms step advancement

SessionStorage XOR decryption (`WO_2024_CHALLENGE` key) serves as last-resort code extraction fallback.

## Tests

```bash
uv run pytest tests/ -v --cov=src/webnav --cov-report=term-missing
# 213 tests, 87.5% coverage
```

## Project Structure

```
src/webnav/
  agent.py        # Orchestrator loop
  browser.py      # Playwright controller
  perception.py   # A11y tree extraction + compression
  page_cleaner.py # Popup/overlay removal
  dispatcher.py   # Tier 0 regex patterns + challenge JS
  solver.py       # Tier 1/2 LLM interface (OpenRouter)
  executor.py     # Action execution (click, scroll, drag, canvas, etc.)
  extractor.py    # Code extraction from page
  state.py        # Step tracking + time budget
  metrics.py      # Per-step metrics + report
```
