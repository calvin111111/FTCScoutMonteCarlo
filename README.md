# FTCScoutMonteCarlo

Monte Carlo simulation that uses [FTCScout](https://ftcscout.org) data to predict FTC match outcomes and rankings.

## How It Works

1. **Fetches team OPR statistics** from the FTCScout GraphQL/REST API (total, auto, teleop, endgame)
2. **Models each team's score** as a normal distribution centered on their OPR
3. **Simulates thousands of events** — sampling scores for every match
4. **Aggregates results** into win probabilities, expected scores, and full ranking distributions

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `requests`, `numpy`, `tabulate`

## Usage

### Predict from a live FTCScout event

```bash
# Predict matches for an event (season year + event code from FTCScout URL)
python main.py --season 2024 --event USNCMP2DV2

# More simulations for higher accuracy
python main.py --season 2024 --event USMOTC --sims 50000

# Set a seed for reproducible results
python main.py --season 2024 --event USNCMP2DV2 --seed 42
```

### Predict from a CSV schedule

```bash
python main.py --season 2024 --csv example_schedule.csv
```

CSV format (header required):
```
match_id,match_name,red1,red2,blue1,blue2
1,Q-1,16461,7244,11115,8393
2,Q-2,9971,14203,16461,5484
```

### Export results to JSON

```bash
python main.py --season 2024 --event USNCMP2DV2 --json results.json
```

### All options

```
--season INT       FTC season year (required)
--event CODE       FTCScout event code
--csv PATH         Path to CSV schedule file
--sims INT         Number of simulations (default: 10000)
--seed INT         Random seed for reproducibility
--no-rankings      Skip ranking predictions
--json PATH        Export results to JSON
--cache-ttl INT    Cache TTL in seconds (default: 3600)
--request-delay FLOAT  Min seconds between API requests (default: 1.0)
```

## Output

**Match Predictions** — per-match win/loss/tie probabilities and expected scores:

```
Match    Red Teams      Blue Teams     Red Win%  Blue Win%  Tie%   Avg Red     Avg Blue
Q-1      16461, 7244    11115, 8393     62.3%     36.5%     1.2%  145 ±28     132 ±25
Q-2      9971, 14203    16461, 5484     41.8%     57.0%     1.2%  128 ±26     138 ±27
```

**Predicted Rankings** — ranking point averages, tie-breaker points, and rank distributions:

```
Pred Rank  Team   Avg RP  Avg TBP   W    L    T    Top 4%  Rank Dist
#1         16461   4.82    412     2.4  0.5  0.1   95.2%   #1:48%, #2:31%, #3:14%
#2         11115   4.15    389     2.1  0.8  0.1   87.3%   #2:29%, #1:26%, #3:22%
```

## API Rate Limiting

The tool is designed to be respectful of FTCScout's servers:

- **Request throttling**: 1 second minimum between API calls (configurable via `--request-delay`)
- **Batched GraphQL queries**: Fetches stats for up to 25 teams in a single request using GraphQL aliases
- **Disk caching**: Responses are cached locally for 1 hour (configurable via `--cache-ttl`) in a `cache/` directory
- **Minimal requests**: A typical event prediction requires only ~3-5 API calls total

## Statistical Model

- Each team's per-match contribution is drawn from `N(OPR, σ²)` where `σ = max(OPR × 0.20, 5)`
- Teams with fewer than 5 matches use a wider distribution (`σ = max(OPR × 0.30, 5)`) to reflect uncertainty
- Alliance scores are the sum of individual team scores (clamped to ≥ 0)
- Rankings use FTC rules: 2 RP for win, 1 for tie, 0 for loss; TBP as tiebreaker

## Data Source

All team statistics come from [FTCScout](https://ftcscout.org) ([API docs](https://ftcscout.org/api/rest), [GitHub](https://github.com/ftc-scout/ftc-scout)).
