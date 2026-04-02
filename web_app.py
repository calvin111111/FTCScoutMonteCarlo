#!/usr/bin/env python3
"""
Flask web interface for FTC Monte Carlo Match Predictor.

Provides a browser-based UI using plain HTML forms (no JavaScript required).
All computation runs server-side — just open the page and submit the form.

Usage:
    python web_app.py
    # Then open http://localhost:5000 in your browser
"""

import sys
import os
import traceback

from flask import Flask, render_template, request

import game_config as gc
from ftcscout_api import FTCScoutAPI
from simulation import MonteCarloSimulator, TeamStats
from kalman_opr import KalmanState
from schedule_parser import parse_ftcscout_matches, parse_csv_schedule

app = Flask(__name__)


def _rp_thresholds(event_type: str) -> tuple:
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


def build_team_stats(api, team_numbers, season):
    kalman_states = api.compute_kalman_opr_for_teams(team_numbers, season)
    stats = {}
    for num in team_numbers:
        ks = kalman_states.get(num, {})
        match_count = 0
        if ks:
            sample_state = next(iter(ks.values()), None)
            match_count = sample_state.match_count if sample_state else 0
        stats[num] = TeamStats(number=num, match_count=match_count, kalman=ks)
    return stats


def run_simulation(season, event_code, csv_text, num_sims, seed, event_type, show_rankings):
    """Run simulation and return (match_results, ranking_dists, info_msg) or raise."""
    api = FTCScoutAPI(cache_ttl=3600, request_delay=0.5)

    if event_code:
        matches_data = api.get_event_matches(season, event_code)
        schedule = parse_ftcscout_matches(matches_data)
        if not schedule:
            raise ValueError(f"No matches found for event '{event_code}' in season {season}.")
    elif csv_text:
        schedule = parse_csv_schedule(csv_text)
        if not schedule:
            raise ValueError("No matches parsed from CSV. Check format: match_id,match_name,red1,red2,blue1,blue2")
    else:
        raise ValueError("Provide either an event code or CSV schedule.")

    team_numbers = sorted({t for m in schedule for t in m.red_teams + m.blue_teams})
    stats = build_team_stats(api, team_numbers, season)

    covered = sum(1 for s in stats.values() if s.match_count > 0)
    no_data = len(team_numbers) - covered

    goal_rp, pattern_rp, movement_rp = _rp_thresholds(event_type)

    sim = MonteCarloSimulator(
        stats, schedule, num_sims, seed,
        goal_rp_threshold=goal_rp,
        pattern_rp_threshold=pattern_rp,
        movement_rp_threshold=movement_rp,
    )
    match_results, ranking_dists = sim.run()

    info = (
        f"Simulated {num_sims:,} runs for {len(schedule)} matches | "
        f"{len(team_numbers)} teams ({covered} with data, {no_data} using defaults) | "
        f"Event type: {event_type}"
    )

    return match_results, ranking_dists, info


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # Collect form data
    season = request.form.get("season", "2025").strip()
    event_code = request.form.get("event_code", "").strip()
    csv_text = request.form.get("csv_schedule", "").strip()
    num_sims = request.form.get("num_sims", "10000").strip()
    seed = request.form.get("seed", "").strip()
    event_type = request.form.get("event_type", "qualifier")
    show_rankings = "show_rankings" in request.form

    # Validate inputs
    errors = []
    try:
        season_int = int(season)
    except ValueError:
        errors.append("Season must be a number (e.g. 2025).")
        season_int = 2025

    try:
        num_sims_int = int(num_sims)
        if num_sims_int < 100:
            errors.append("Minimum 100 simulations.")
            num_sims_int = 100
        elif num_sims_int > 100000:
            errors.append("Maximum 100,000 simulations.")
            num_sims_int = 100000
    except ValueError:
        errors.append("Number of simulations must be a number.")
        num_sims_int = 10000

    seed_int = None
    if seed:
        try:
            seed_int = int(seed)
        except ValueError:
            errors.append("Seed must be a number.")

    if not event_code and not csv_text:
        errors.append("Provide either an event code or a CSV schedule.")

    if errors:
        return render_template("index.html", errors=errors,
                               season=season, event_code=event_code,
                               csv_schedule=csv_text, num_sims=num_sims,
                               seed=seed, event_type=event_type,
                               show_rankings=show_rankings)

    # Run simulation
    try:
        match_results, ranking_dists, info = run_simulation(
            season_int, event_code or None, csv_text or None,
            num_sims_int, seed_int, event_type, show_rankings,
        )
    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", errors=[str(e)],
                               season=season, event_code=event_code,
                               csv_schedule=csv_text, num_sims=num_sims,
                               seed=seed, event_type=event_type,
                               show_rankings=show_rankings)

    # Prepare ranking data
    sorted_rankings = []
    total_sims = num_sims_int
    if show_rankings and ranking_dists:
        total_sims = sum(next(iter(ranking_dists.values())).rank_counts.values())
        for i, d in enumerate(sorted(ranking_dists.values(), key=lambda x: x.avg_rank), 1):
            top4 = sum(d.rank_counts.get(r, 0) for r in range(1, 5))
            top4_pct = top4 / total_sims * 100
            top_ranks = sorted(d.rank_counts.items(), key=lambda x: -x[1])[:3]
            rank_dist = ", ".join(f"#{r}:{c / total_sims * 100:.0f}%" for r, c in top_ranks)
            sorted_rankings.append({
                "pred_rank": i,
                "number": d.number,
                "avg_rp": f"{d.avg_rp:.2f}",
                "avg_tbp": f"{d.avg_tbp:.0f}",
                "wins": f"{d.win_count:.1f}",
                "losses": f"{d.loss_count:.1f}",
                "ties": f"{d.tie_count:.1f}",
                "top4_pct": f"{top4_pct:.1f}%",
                "rank_dist": rank_dist,
            })

    return render_template("results.html",
                           info=info,
                           match_results=match_results,
                           rankings=sorted_rankings,
                           show_rankings=show_rankings,
                           season=season,
                           event_code=event_code,
                           csv_schedule=csv_text,
                           num_sims=num_sims,
                           seed=seed,
                           event_type=event_type)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
