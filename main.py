#!/usr/bin/env python3
"""
FTC Monte Carlo Match Predictor

Fetches team statistics from FTCScout and runs Monte Carlo simulations
to predict match outcomes and final rankings for an FTC event.

Usage:
    # Predict from a live FTCScout event
    python main.py --season 2024 --event USNCMP2DV2

    # Predict from a CSV schedule file
    python main.py --season 2024 --csv schedule.csv

    # Adjust simulation parameters
    python main.py --season 2024 --event USMOTC --sims 50000 --seed 42
"""

import argparse
import sys
import json

from ftcscout_api import FTCScoutAPI
from simulation import MonteCarloSimulator, TeamStats, ScheduledMatch
from schedule_parser import parse_ftcscout_matches, parse_csv_file
from output import format_match_predictions, format_rankings


def build_team_stats_from_api(api: FTCScoutAPI, team_numbers: list[int],
                              season: int) -> dict[int, TeamStats]:
    """Fetch QuickStats for all teams and build TeamStats objects."""
    print(f"Fetching QuickStats for {len(team_numbers)} teams "
          f"(season {season})...")
    raw = api.get_quick_stats_for_teams(team_numbers, season)

    stats = {}
    missing = []
    for num in team_numbers:
        qs = raw.get(num)
        if qs and qs.get("tot"):
            stats[num] = TeamStats(
                number=num,
                opr_total=qs["tot"]["value"] or 0,
                opr_auto=qs["auto"]["value"] or 0,
                opr_teleop=qs["dc"]["value"] or 0,
                opr_endgame=qs["eg"]["value"] or 0,
                match_count=qs.get("count", 0),
            )
        else:
            missing.append(num)

    if missing:
        print(f"  Warning: No QuickStats found for {len(missing)} team(s): "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
        print(f"  These teams will use default (0 OPR) values.")

    return stats


def run_event_prediction(api: FTCScoutAPI, season: int, event_code: str,
                         num_sims: int, seed: int | None,
                         show_rankings: bool):
    """Full pipeline: fetch event data, simulate, print results."""
    print(f"\nFetching event data for {event_code} (season {season})...")
    matches_data = api.get_event_matches(season, event_code)
    print(f"  Found {len(matches_data)} matches.")

    schedule = parse_ftcscout_matches(matches_data)
    if not schedule:
        print("No matches found in schedule. Check event code and season.")
        sys.exit(1)
    print(f"  Parsed {len(schedule)} matches for simulation.")

    # Collect teams
    team_numbers = set()
    for m in schedule:
        team_numbers.update(m.red_teams)
        team_numbers.update(m.blue_teams)
    team_numbers = sorted(team_numbers)
    print(f"  {len(team_numbers)} teams in schedule.")

    stats = build_team_stats_from_api(api, team_numbers, season)

    print(f"\nRunning {num_sims:,} simulations...")
    sim = MonteCarloSimulator(stats, schedule, num_sims, seed)
    match_results, ranking_dists = sim.run()
    print("Done!\n")

    print("=" * 80)
    print("MATCH PREDICTIONS")
    print("=" * 80)
    print(format_match_predictions(match_results))

    if show_rankings:
        print("\n")
        print("=" * 80)
        print("PREDICTED RANKINGS")
        print("=" * 80)
        print(format_rankings(ranking_dists))

    return match_results, ranking_dists


def run_csv_prediction(api: FTCScoutAPI, season: int, csv_path: str,
                       num_sims: int, seed: int | None,
                       show_rankings: bool):
    """Run prediction from a CSV schedule file."""
    print(f"\nLoading schedule from {csv_path}...")
    schedule = parse_csv_file(csv_path)
    if not schedule:
        print("No matches found in CSV. Check format.")
        sys.exit(1)
    print(f"  Parsed {len(schedule)} matches.")

    team_numbers = set()
    for m in schedule:
        team_numbers.update(m.red_teams)
        team_numbers.update(m.blue_teams)
    team_numbers = sorted(team_numbers)
    print(f"  {len(team_numbers)} teams in schedule.")

    stats = build_team_stats_from_api(api, team_numbers, season)

    print(f"\nRunning {num_sims:,} simulations...")
    sim = MonteCarloSimulator(stats, schedule, num_sims, seed)
    match_results, ranking_dists = sim.run()
    print("Done!\n")

    print("=" * 80)
    print("MATCH PREDICTIONS")
    print("=" * 80)
    print(format_match_predictions(match_results))

    if show_rankings:
        print("\n")
        print("=" * 80)
        print("PREDICTED RANKINGS")
        print("=" * 80)
        print(format_rankings(ranking_dists))

    return match_results, ranking_dists


def export_json(match_results, ranking_dists, path: str):
    """Export results to JSON."""
    data = {
        "matches": [
            {
                "match_name": r.match.match_name,
                "red_teams": r.match.red_teams,
                "blue_teams": r.match.blue_teams,
                "red_win_pct": round(r.red_win_pct, 2),
                "blue_win_pct": round(r.blue_win_pct, 2),
                "tie_pct": round(r.tie_pct, 2),
                "avg_red_score": round(r.avg_red_score, 1),
                "avg_blue_score": round(r.avg_blue_score, 1),
            }
            for r in match_results
        ],
        "rankings": [
            {
                "team": d.number,
                "avg_rank": round(d.avg_rank, 2),
                "avg_rp": round(d.avg_rp, 2),
                "avg_tbp": round(d.avg_tbp, 1),
                "wins": round(d.win_count, 2),
                "losses": round(d.loss_count, 2),
                "ties": round(d.tie_count, 2),
            }
            for d in sorted(ranking_dists.values(), key=lambda x: x.avg_rank)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="FTC Monte Carlo Match Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --season 2024 --event USNCMP2DV2
  python main.py --season 2024 --event USMOTC --sims 50000
  python main.py --season 2024 --csv schedule.csv --seed 42
  python main.py --season 2024 --event USNCMP2DV2 --json results.json
        """,
    )
    parser.add_argument("--season", type=int, required=True,
                        help="FTC season year (e.g. 2024 for INTO THE DEEP)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", type=str,
                        help="FTCScout event code (e.g. USNCMP2DV2)")
    source.add_argument("--csv", type=str,
                        help="Path to CSV schedule file")
    parser.add_argument("--sims", type=int, default=10000,
                        help="Number of simulations (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-rankings", action="store_true",
                        help="Skip ranking predictions")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--cache-ttl", type=int, default=3600,
                        help="Cache TTL in seconds (default: 3600)")
    parser.add_argument("--request-delay", type=float, default=1.0,
                        help="Min seconds between API requests (default: 1.0)")

    args = parser.parse_args()

    api = FTCScoutAPI(cache_ttl=args.cache_ttl,
                      request_delay=args.request_delay)

    if args.event:
        match_results, ranking_dists = run_event_prediction(
            api, args.season, args.event, args.sims, args.seed,
            not args.no_rankings)
    else:
        match_results, ranking_dists = run_csv_prediction(
            api, args.season, args.csv, args.sims, args.seed,
            not args.no_rankings)

    if args.json:
        export_json(match_results, ranking_dists, args.json)


if __name__ == "__main__":
    main()
