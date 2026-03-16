#!/usr/bin/env python3
"""
FTC Monte Carlo Match Predictor — DECODE 2025-2026

Fetches per-match score breakdowns from FTCScout, computes per-category OPR
using a Kalman filter, and runs Monte Carlo simulations to predict match
outcomes and final rankings for an FTC event.

Penalty points are applied to the OPPOSING alliance (not the team that commits
them), matching the actual FTC scoring rules.

Bonus RP thresholds (movementRP, goalRP, patternRP) are configurable via
game_config.py — verify against the official DECODE Competition Manual
(Table 10-3, Section 10.5.4) for your event tier.

Usage:
    # Predict from a live FTCScout event (qualifying)
    python main.py --season 2025 --event NJCMP

    # Predict for a regional championship (higher RP thresholds)
    python main.py --season 2025 --event USNCMP2DV2 --event-type regional

    # Predict from a CSV schedule file
    python main.py --season 2025 --csv schedule.csv

    # Adjust simulation parameters
    python main.py --season 2025 --event USMOTC --sims 50000 --seed 42
"""

import argparse
import sys
import json

import game_config as gc
from ftcscout_api import FTCScoutAPI
from simulation import MonteCarloSimulator, TeamStats, ScheduledMatch
from kalman_opr import KalmanState
from schedule_parser import parse_ftcscout_matches, parse_csv_file
from output import format_match_predictions, format_rankings


def build_team_stats_from_kalman(
    api: FTCScoutAPI,
    team_numbers: list[int],
    season: int,
) -> dict[int, TeamStats]:
    """Fetch per-match category breakdowns and compute Kalman OPR for all teams.

    Returns a mapping of team number -> TeamStats with Kalman OPR states
    for each scoring category.  Teams with no match history get prior-only
    states (mean = 0, high variance).
    """
    print(f"Fetching per-match score breakdowns for {len(team_numbers)} teams "
          f"(season {season})...")
    kalman_states = api.compute_kalman_opr_for_teams(team_numbers, season)

    stats: dict[int, TeamStats] = {}
    for num in team_numbers:
        ks = kalman_states.get(num, {})
        match_count = 0
        if ks:
            # Use any category's match_count (they're all incremented together)
            sample_state = next(iter(ks.values()), None)
            match_count = sample_state.match_count if sample_state else 0

        stats[num] = TeamStats(
            number=num,
            match_count=match_count,
            kalman=ks,
        )

    covered   = sum(1 for s in stats.values() if s.match_count > 0)
    no_data   = len(team_numbers) - covered
    print(f"  Kalman OPR computed for {covered}/{len(team_numbers)} teams "
          f"({no_data} teams use prior defaults — no match history yet).")
    return stats


def _rp_thresholds(event_type: str) -> tuple[float, float, float]:
    """Return (goal_rp, pattern_rp, movement_rp) thresholds for the event type."""
    if event_type == "regional":
        return (
            gc.GOAL_RP_CLASSIFIED_PTS_REGIONAL,
            gc.PATTERN_RP_PTS_REGIONAL,
            gc.MOVEMENT_RP_THRESHOLD,
        )
    return (
        gc.GOAL_RP_CLASSIFIED_PTS_QUALIFIER,
        gc.PATTERN_RP_PTS_QUALIFIER,
        gc.MOVEMENT_RP_THRESHOLD,
    )


def run_event_prediction(
    api: FTCScoutAPI,
    season: int,
    event_code: str,
    num_sims: int,
    seed: int | None,
    show_rankings: bool,
    event_type: str,
):
    """Full pipeline: fetch event data, compute Kalman OPR, simulate, print."""
    print(f"\nFetching event data for {event_code} (season {season})...")
    matches_data = api.get_event_matches(season, event_code)
    print(f"  Found {len(matches_data)} matches.")

    schedule = parse_ftcscout_matches(matches_data)
    if not schedule:
        print("No matches found in schedule. Check event code and season.")
        sys.exit(1)
    print(f"  Parsed {len(schedule)} matches for simulation.")

    team_numbers = sorted({t for m in schedule for t in m.red_teams + m.blue_teams})
    print(f"  {len(team_numbers)} teams in schedule.")

    stats = build_team_stats_from_kalman(api, team_numbers, season)

    goal_rp, pattern_rp, movement_rp = _rp_thresholds(event_type)
    print(f"\nRP thresholds ({event_type}):")
    print(f"  movementRP : auto_leave_pts >= {movement_rp}")
    print(f"  goalRP     : classified_pts / {gc.CLASSIFIED_ARTIFACT_PTS} >= "
          f"{goal_rp / gc.CLASSIFIED_ARTIFACT_PTS:.0f} artifacts "
          f"(= {goal_rp:.0f} pts)")
    print(f"  patternRP  : pattern_pts >= {pattern_rp}")

    print(f"\nRunning {num_sims:,} simulations...")
    sim = MonteCarloSimulator(
        stats, schedule, num_sims, seed,
        goal_rp_threshold=goal_rp,
        pattern_rp_threshold=pattern_rp,
        movement_rp_threshold=movement_rp,
    )
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


def run_csv_prediction(
    api: FTCScoutAPI,
    season: int,
    csv_path: str,
    num_sims: int,
    seed: int | None,
    show_rankings: bool,
    event_type: str,
):
    """Run prediction from a CSV schedule file."""
    print(f"\nLoading schedule from {csv_path}...")
    schedule = parse_csv_file(csv_path)
    if not schedule:
        print("No matches found in CSV. Check format.")
        sys.exit(1)
    print(f"  Parsed {len(schedule)} matches.")

    team_numbers = sorted({t for m in schedule for t in m.red_teams + m.blue_teams})
    print(f"  {len(team_numbers)} teams in schedule.")

    stats = build_team_stats_from_kalman(api, team_numbers, season)

    goal_rp, pattern_rp, movement_rp = _rp_thresholds(event_type)
    print(f"\nRP thresholds ({event_type}):")
    print(f"  movementRP : auto_leave_pts >= {movement_rp}")
    print(f"  goalRP     : classified_pts / {gc.CLASSIFIED_ARTIFACT_PTS} >= "
          f"{goal_rp / gc.CLASSIFIED_ARTIFACT_PTS:.0f} artifacts "
          f"(= {goal_rp:.0f} pts)")
    print(f"  patternRP  : pattern_pts >= {pattern_rp}")

    print(f"\nRunning {num_sims:,} simulations...")
    sim = MonteCarloSimulator(
        stats, schedule, num_sims, seed,
        goal_rp_threshold=goal_rp,
        pattern_rp_threshold=pattern_rp,
        movement_rp_threshold=movement_rp,
    )
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
                "match_name":    r.match.match_name,
                "red_teams":     r.match.red_teams,
                "blue_teams":    r.match.blue_teams,
                "red_win_pct":   round(r.red_win_pct,   2),
                "blue_win_pct":  round(r.blue_win_pct,  2),
                "tie_pct":       round(r.tie_pct,        2),
                "avg_red_score": round(r.avg_red_score,  1),
                "avg_blue_score":round(r.avg_blue_score, 1),
            }
            for r in match_results
        ],
        "rankings": [
            {
                "team":      d.number,
                "avg_rank":  round(d.avg_rank,  2),
                "avg_rp":    round(d.avg_rp,    2),
                "avg_tbp1":  round(d.avg_tbp,   1),
                "avg_tbp2":  round(d.avg_tbp2,  1),
                "wins":      round(d.win_count,  2),
                "losses":    round(d.loss_count, 2),
                "ties":      round(d.tie_count,  2),
            }
            for d in sorted(ranking_dists.values(), key=lambda x: x.avg_rank)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="FTC Monte Carlo Match Predictor (DECODE 2025-2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --season 2025 --event NJCMP
  python main.py --season 2025 --event USNCMP2DV2 --event-type regional
  python main.py --season 2025 --event USMOTC --sims 50000
  python main.py --season 2025 --csv schedule.csv --seed 42
  python main.py --season 2025 --event USNCMP2DV2 --json results.json

Event types:
  qualifier (default) — league meets, qualifying tournaments
  regional            — regional championships (RCMP); uses higher RP thresholds
        """,
    )
    parser.add_argument("--season", type=int, required=True,
                        help="FTC season year (2025 for DECODE)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", type=str,
                        help="FTCScout event code (e.g. NJCMP)")
    source.add_argument("--csv", type=str,
                        help="Path to CSV schedule file")
    parser.add_argument("--event-type", type=str, default="qualifier",
                        choices=["qualifier", "regional"],
                        help="Event tier — controls RP thresholds (default: qualifier)")
    parser.add_argument("--sims", type=int, default=10_000,
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
            not args.no_rankings, args.event_type,
        )
    else:
        match_results, ranking_dists = run_csv_prediction(
            api, args.season, args.csv, args.sims, args.seed,
            not args.no_rankings, args.event_type,
        )

    if args.json:
        export_json(match_results, ranking_dists, args.json)


if __name__ == "__main__":
    main()
