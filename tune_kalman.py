#!/usr/bin/env python3
"""
Tune Kalman filter hyperparameters from past FTC season data.

For each past season this script:
  1. Fetches match scores from as many events as requested.
  2. Runs a two-pass per-event calibration (mirrors _calibrate_kalman_params):
       Pass 1 — rough 1-D Kalman to get initial OPR estimates.
       Pass 2 — compute residuals  (alliance_score − opr1 − opr2),
                use their variance as the per-event noise floor.
  3. Grid-searches three dimensionless hyperparameters that apply the same
     way to every event and every season:
         R_ratio        — final R = R_ratio × var(pass-2 residuals)
         Q_fraction     — process noise  Q = Q_fraction × R  per match
         prior_var_fac  — initial prior_var = prior_var_fac × R
  4. Writes the best values back to game_config.py as:
         KALMAN_TUNED_R_RATIO
         KALMAN_TUNED_Q_FRACTION
         KALMAN_TUNED_PRIOR_VAR_FACTOR

Using past seasons keeps current-event match data completely out of the
tuning loop, giving unbiased hyperparameter estimates.

Usage:
    python tune_kalman.py
    python tune_kalman.py --seasons 2024 2023 --max-events 20
    python tune_kalman.py --seasons 2024 --no-save          # dry run
    python tune_kalman.py --seasons 2024 --cache-ttl 86400  # 1-day cache
"""

import argparse
import math
import os
import re
from collections import defaultdict

import game_config as gc
from ftcscout_api import FTCScoutAPI


# ---------------------------------------------------------------------------
# Event discovery
# ---------------------------------------------------------------------------

def _get_event_codes(api: FTCScoutAPI, season: int) -> list[str]:
    """Return event codes for a past season.

    Tries the REST events endpoint first, falls back to GraphQL eventsSearch.
    Returns an empty list if neither succeeds so the caller can skip gracefully.
    """
    # ---- REST ---------------------------------------------------------------
    try:
        data = api._rest_get(f"/events?season={season}")
        if isinstance(data, list) and data:
            codes = [e.get("code") for e in data if isinstance(e, dict) and e.get("code")]
            if codes:
                return codes
    except Exception:
        pass

    # ---- GraphQL fallback ---------------------------------------------------
    query = """
    query($season: Int!) {
        eventsSearch(season: $season) {
            code
        }
    }
    """
    try:
        data = api._graphql(query, {"season": season})
        events = data.get("eventsSearch") or []
        codes = [e.get("code") for e in events if isinstance(e, dict) and e.get("code")]
        if codes:
            return codes
    except Exception:
        pass

    return []


# ---------------------------------------------------------------------------
# Match observation extraction
# ---------------------------------------------------------------------------

def _extract_alliance_obs(matches: list[dict]) -> list[tuple]:
    """Return sorted list of (match_num, t1, t2, alliance_total_score).

    One tuple per played alliance (two per match).  Skips matches without
    a parseable score or fewer than 2 teams on either side.
    """
    obs = []
    for match in matches:
        if not match.get("hasBeenPlayed", False):
            continue

        teams    = match.get("teams", [])
        sort_key = match.get("matchNum", 0)

        red_teams  = [t["teamNumber"] for t in teams
                      if isinstance(t, dict) and t.get("alliance") == "Red"]
        blue_teams = [t["teamNumber"] for t in teams
                      if isinstance(t, dict) and t.get("alliance") == "Blue"]

        scores = match.get("scores") or {}
        red_score = (
            match.get("redScore")
            or (scores.get("red") or {}).get("totalPoints")
            or (scores.get("Red") or {}).get("totalPoints")
        )
        blue_score = (
            match.get("blueScore")
            or (scores.get("blue") or {}).get("totalPoints")
            or (scores.get("Blue") or {}).get("totalPoints")
        )

        if len(red_teams) == 2 and red_score is not None:
            obs.append((sort_key, red_teams[0], red_teams[1], float(red_score)))
        if len(blue_teams) == 2 and blue_score is not None:
            obs.append((sort_key, blue_teams[0], blue_teams[1], float(blue_score)))

    obs.sort()
    return obs


# ---------------------------------------------------------------------------
# 1-D Kalman OPR helpers
# ---------------------------------------------------------------------------

def _run_kalman_1d(
    obs:        list[tuple],
    R:          float,
    Q:          float,
    prior_mean: float,
    prior_var:  float,
) -> dict[int, float]:
    """Single sequential pass of 1-D Kalman OPR.  Returns {team: final_mean}."""
    states: dict[int, list] = defaultdict(lambda: [prior_mean, prior_var])

    for (_, t1, t2, score) in obs:
        s1, s2 = states[t1], states[t2]
        innov  = score - (s1[0] + s2[0])
        S      = s1[1] + s2[1] + R
        K1, K2 = s1[1] / S, s2[1] / S

        s1[0] += K1 * innov
        s1[1]  = max((1.0 - K1) * s1[1] + Q, 1e-9)
        s2[0] += K2 * innov
        s2[1]  = max((1.0 - K2) * s2[1] + Q, 1e-9)

    return {t: v[0] for t, v in states.items()}


def _residual_var(
    obs:        list[tuple],
    rough_means: dict[int, float],
    prior_mean: float,
) -> float:
    """Compute variance of residuals (alliance_score − opr1 − opr2)."""
    residuals = [
        score - (rough_means.get(t1, prior_mean) + rough_means.get(t2, prior_mean))
        for (_, t1, t2, score) in obs
    ]
    if len(residuals) < 5:
        return 1.0
    mu  = sum(residuals) / len(residuals)
    var = sum((r - mu) ** 2 for r in residuals) / len(residuals)
    return max(var, 0.25)


# ---------------------------------------------------------------------------
# Per-event evaluation  (two-pass, consistent with _calibrate_kalman_params)
# ---------------------------------------------------------------------------

def _evaluate_event(
    obs:             list[tuple],
    R_ratio:         float,
    Q_fraction:      float,
    prior_var_fac:   float,
) -> list[float]:
    """Return per-match squared prediction errors for one event.

    Mirrors the two-pass logic in FTCScoutAPI._calibrate_kalman_params:
      1. Rough Kalman (R = raw_score_var / 2, large prior_var, Q=0) to get
         initial team estimates.
      2. Compute residuals; derive per-event residual_var.
      3. Evaluate with:  R = R_ratio × residual_var
                         Q = Q_fraction × R
                         prior_var = prior_var_fac × R
    using a leave-one-out sequential Kalman update (prediction before update).
    """
    if len(obs) < 10:
        return []

    all_scores = [s for (_, _, _, s) in obs]
    prior_mean = sum(all_scores) / len(all_scores) / 2.0
    score_var  = sum((s - 2.0 * prior_mean) ** 2 for s in all_scores) / len(all_scores)

    # ---- Pass 1: rough Kalman to seed OPR estimates -----
    rough_R     = max(score_var / 2.0, 1.0)
    rough_pvar  = max(2500.0, 10.0 * rough_R)
    rough_means = _run_kalman_1d(obs, rough_R, Q=0.0, prior_mean=prior_mean,
                                 prior_var=rough_pvar)

    # ---- Pass 2: residual-based R -----------------------
    res_var   = _residual_var(obs, rough_means, prior_mean)
    R         = R_ratio    * res_var
    Q         = Q_fraction * R
    prior_var = prior_var_fac * R

    # ---- Sequential LOO evaluation ----------------------
    states: dict[int, list] = defaultdict(lambda: [prior_mean, prior_var])
    sq_errors: list[float]  = []

    for (_, t1, t2, score) in obs:
        s1, s2 = states[t1], states[t2]

        # Predict BEFORE updating this match
        sq_errors.append((score - (s1[0] + s2[0])) ** 2)

        innov  = score - (s1[0] + s2[0])
        S      = s1[1] + s2[1] + R
        K1, K2 = s1[1] / S, s2[1] / S
        s1[0] += K1 * innov
        s1[1]  = max((1.0 - K1) * s1[1] + Q, 1e-9)
        s2[0] += K2 * innov
        s2[1]  = max((1.0 - K2) * s2[1] + Q, 1e-9)

    return sq_errors


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

R_RATIO_GRID      = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
Q_FRACTION_GRID   = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
PRIOR_VAR_GRID    = [5.0, 10.0, 20.0, 50.0, 100.0]


def grid_search(
    all_event_obs: list[list[tuple]],
    r_ratios:    list[float] = R_RATIO_GRID,
    q_fractions: list[float] = Q_FRACTION_GRID,
    pv_factors:  list[float] = PRIOR_VAR_GRID,
) -> tuple[float, float, float, float]:
    """Find (R_ratio, Q_fraction, prior_var_fac, best_rmse) via grid search."""
    n_combos = len(r_ratios) * len(q_fractions) * len(pv_factors)
    print(f"  Searching {n_combos} hyperparameter combinations...")

    best_rmse   = float("inf")
    best_params = (1.0, 0.1, 10.0)

    for r_ratio in r_ratios:
        for q_frac in q_fractions:
            for pv_fac in pv_factors:
                all_sq: list[float] = []
                for obs in all_event_obs:
                    all_sq.extend(_evaluate_event(obs, r_ratio, q_frac, pv_fac))
                if not all_sq:
                    continue
                rmse = math.sqrt(sum(all_sq) / len(all_sq))
                if rmse < best_rmse:
                    best_rmse   = rmse
                    best_params = (r_ratio, q_frac, pv_fac)

    return (*best_params, best_rmse)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_season(
    api:        FTCScoutAPI,
    season:     int,
    max_events: int,
) -> list[list[tuple]]:
    """Fetch and return per-event alliance observation lists for one season."""
    print(f"  Fetching events for season {season}...")
    codes = _get_event_codes(api, season)
    if not codes:
        print(f"    No events found for season {season} — skipping.")
        return []
    print(f"    Discovered {len(codes)} events; using up to {max_events}.")

    result: list[list[tuple]] = []
    for code in codes:
        if len(result) >= max_events:
            break
        try:
            matches = api.get_event_matches(season, code)
        except Exception as exc:
            print(f"    {code}: skipped ({exc})")
            continue

        obs = _extract_alliance_obs(matches)
        if len(obs) < 20:          # skip very small / unsanctioned events
            continue

        result.append(obs)
        n_teams = len({t for (_, t1, t2, _) in obs for t in (t1, t2)})
        print(f"    {code}: {len(obs):>4} alliance obs  {n_teams:>3} teams")

    return result


# ---------------------------------------------------------------------------
# game_config.py writer
# ---------------------------------------------------------------------------

def _replace_constant(text: str, name: str, value: float) -> str:
    """Replace  NAME: float = <old>  with  NAME: float = <new>  in text."""
    pattern     = rf"^({re.escape(name)}\s*:\s*float\s*=\s*)[\d.eE+\-]+(.*)$"
    replacement = rf"\g<1>{value!r}\2"
    new_text    = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    if new_text == text:
        raise ValueError(f"Could not find '{name}: float = ...' in game_config.py")
    return new_text


def write_game_config(
    path:      str,
    r_ratio:   float,
    q_frac:    float,
    pv_factor: float,
) -> None:
    with open(path) as f:
        text = f.read()
    text = _replace_constant(text, "KALMAN_TUNED_R_RATIO",          r_ratio)
    text = _replace_constant(text, "KALMAN_TUNED_Q_FRACTION",       q_frac)
    text = _replace_constant(text, "KALMAN_TUNED_PRIOR_VAR_FACTOR", pv_factor)
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune Kalman filter hyperparameters from past FTC season data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune_kalman.py
  python tune_kalman.py --seasons 2024 2023 --max-events 20
  python tune_kalman.py --seasons 2024 --no-save   # dry run, no file update
        """,
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2024, 2023],
        help="Past FTC seasons to tune on (default: 2024 2023)",
    )
    parser.add_argument(
        "--max-events", type=int, default=15,
        help="Maximum events to use per season (default: 15)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print best params but do not write to game_config.py",
    )
    parser.add_argument(
        "--cache-ttl", type=int, default=86400,
        help="API cache TTL in seconds (default: 86400 = 1 day)",
    )
    args = parser.parse_args()

    api = FTCScoutAPI(cache_ttl=args.cache_ttl)

    all_obs: list[list[tuple]] = []
    for season in args.seasons:
        print(f"\nSeason {season}:")
        season_obs = collect_season(api, season, args.max_events)
        all_obs.extend(season_obs)

    if not all_obs:
        print("\nNo data collected — cannot tune.  Check --seasons values and network.")
        return

    total_alliance_obs = sum(len(obs) for obs in all_obs)
    print(
        f"\nCollected {len(all_obs)} events  "
        f"({total_alliance_obs} alliance observations)."
    )

    print("\nRunning grid search...")
    r_ratio, q_frac, pv_factor, rmse = grid_search(all_obs)

    print(f"\nBest hyperparameters  (LOO RMSE = {rmse:.2f} pts):")
    print(f"  KALMAN_TUNED_R_RATIO          = {r_ratio}")
    print(f"  KALMAN_TUNED_Q_FRACTION       = {q_frac}")
    print(f"  KALMAN_TUNED_PRIOR_VAR_FACTOR = {pv_factor}")

    if not args.no_save:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "game_config.py")
        write_game_config(config_path, r_ratio, q_frac, pv_factor)
        print(f"\nUpdated {config_path}")
    else:
        print("\n(--no-save: game_config.py not modified)")


if __name__ == "__main__":
    main()
