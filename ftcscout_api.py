"""
FTCScout API client with rate limiting and local caching.

Uses the GraphQL API at api.ftcscout.org/graphql to fetch team statistics,
event data, and match information. Implements request throttling and disk
caching to avoid overwhelming the server.
"""

import json
import math
import os
import time
import hashlib
import requests

from kalman_opr import (
    extract_alliance_breakdown,
    compute_kalman_opr,
    KalmanState,
)
import game_config as gc

GRAPHQL_ENDPOINT = "https://api.ftcscout.org/graphql"
REST_ENDPOINT = "https://api.ftcscout.org/rest/v1"

# Minimum seconds between API requests
REQUEST_DELAY = 1.0

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class FTCScoutAPI:
    """Client for the FTCScout API with rate limiting and caching."""

    def __init__(self, cache_dir=CACHE_DIR, request_delay=REQUEST_DELAY,
                 cache_ttl=3600):
        """
        Args:
            cache_dir: Directory to store cached API responses.
            request_delay: Minimum seconds between requests.
            cache_ttl: Seconds before a cached response is considered stale.
        """
        self.cache_dir = cache_dir
        self.request_delay = request_delay
        self.cache_ttl = cache_ttl
        self._last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _throttle(self):
        """Wait if needed to respect the request delay."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _read_cache(self, key: str):
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                entry = json.load(f)
            if time.time() - entry["ts"] > self.cache_ttl:
                return None
            return entry["data"]
        except (json.JSONDecodeError, KeyError):
            return None

    def _write_cache(self, key: str, data):
        path = self._cache_path(key)
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "data": data}, f)

    # ------------------------------------------------------------------
    # GraphQL helpers
    # ------------------------------------------------------------------

    def _graphql(self, query: str, variables: dict = None):
        """Execute a GraphQL query, using cache when available."""
        payload = json.dumps({"query": query, "variables": variables or {}},
                             sort_keys=True)
        key = self._cache_key(payload)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        self._throttle()
        resp = self.session.post(GRAPHQL_ENDPOINT, data=payload)
        resp.raise_for_status()
        result = resp.json()
        if "errors" in result:
            raise RuntimeError(f"GraphQL errors: {result['errors']}")
        self._write_cache(key, result["data"])
        return result["data"]

    def _rest_get(self, path: str):
        """GET from the REST API, using cache when available."""
        key = self._cache_key(path)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        self._throttle()
        resp = self.session.get(f"{REST_ENDPOINT}{path}")
        resp.raise_for_status()
        data = resp.json()
        self._write_cache(key, data)
        return data

    # ------------------------------------------------------------------
    # High-level queries
    # ------------------------------------------------------------------

    def get_team(self, number: int) -> dict:
        """Fetch basic team info."""
        return self._rest_get(f"/teams/{number}")

    def get_team_quick_stats(self, number: int, season: int) -> dict | None:
        """Fetch QuickStats (OPR breakdown) for a team in a season.

        Returns dict with keys: tot, auto, dc, eg  (each has value, rank)
        or None if no data is available.
        """
        query = """
        query($number: Int!, $season: Int!) {
            teamByNumber(number: $number) {
                quickStats(season: $season) {
                    tot { value }
                    auto { value }
                    dc { value }
                    eg { value }
                    count
                }
            }
        }
        """
        data = self._graphql(query, {"number": number, "season": season})
        team = data.get("teamByNumber")
        if not team:
            return None
        return team.get("quickStats")

    def get_event(self, season: int, code: str) -> dict:
        """Fetch event metadata."""
        return self._rest_get(f"/events/{season}/{code}")

    def get_event_teams(self, season: int, code: str) -> list[dict]:
        """Fetch all team participations at an event (includes stats)."""
        return self._rest_get(f"/events/{season}/{code}/teams")

    def get_event_matches(self, season: int, code: str) -> list[dict]:
        """Fetch all matches at an event."""
        return self._rest_get(f"/events/{season}/{code}/matches")

    def get_team_matches(self, number: int, season: int = None,
                         event_code: str = None) -> list[dict]:
        """Fetch all matches for a team, optionally filtered."""
        path = f"/teams/{number}/matches"
        params = []
        if season is not None:
            params.append(f"season={season}")
        if event_code is not None:
            params.append(f"eventCode={event_code}")
        if params:
            path += "?" + "&".join(params)
        return self._rest_get(path)

    def get_team_events(self, number: int, season: int) -> list[dict]:
        """Fetch all event participations for a team in a season."""
        return self._rest_get(f"/teams/{number}/events?season={season}")

    # ------------------------------------------------------------------
    # Bulk fetch with progress — batches teams to minimise requests
    # ------------------------------------------------------------------

    def get_quick_stats_for_teams(self, team_numbers: list[int],
                                  season: int) -> dict[int, dict | None]:
        """Fetch QuickStats for many teams. Returns {number: stats_or_None}.

        Uses a single GraphQL query with aliases to fetch all teams at once,
        drastically reducing the number of API calls.
        """
        results = {}
        # GraphQL alias names must start with a letter
        batch_size = 25  # keep queries small
        batches = [team_numbers[i:i + batch_size]
                   for i in range(0, len(team_numbers), batch_size)]

        for batch in batches:
            parts = []
            for num in batch:
                alias = f"t{num}"
                parts.append(f"""
                    {alias}: teamByNumber(number: {num}) {{
                        quickStats(season: {season}) {{
                            tot {{ value }}
                            auto {{ value }}
                            dc {{ value }}
                            eg {{ value }}
                            count
                        }}
                    }}
                """)
            query = "{ " + "\n".join(parts) + " }"
            data = self._graphql(query)
            for num in batch:
                alias = f"t{num}"
                team_data = data.get(alias)
                if team_data:
                    results[num] = team_data.get("quickStats")
                else:
                    results[num] = None

        return results

    def compute_team_std_devs(self, team_stats: dict, season: int,
                              min_matches: int = 3) -> dict[int, float]:
        """Compute empirical std dev for each team from match-by-match history.

        For each played match, estimates the team's individual contribution as:
            contribution = alliance_score - partner_OPR

        Subtracting the partner's OPR isolates the team's own scoring variance
        from the noise introduced by varying partner quality. The std dev of
        these residuals is used in place of the hardcoded OPR percentage.

        Args:
            team_stats: Mapping of team number -> TeamStats (needs opr_total).
            season: FTC season year.
            min_matches: Minimum played matches required to use empirical std dev.
                         Teams below this threshold keep the OPR-derived fallback.

        Returns:
            Mapping of team number -> empirical std dev (only for teams with
            enough data; others are omitted so the fallback applies).
        """
        result = {}
        for team_num, stats in team_stats.items():
            matches = self.get_team_matches(team_num, season=season)
            contributions = []
            for match in matches:
                if not match.get("hasBeenPlayed", False):
                    continue

                teams = match.get("teams", [])

                # Identify this team's alliance and partner
                my_alliance = None
                partner_num = None
                for t in teams:
                    if t.get("teamNumber") == team_num:
                        my_alliance = t.get("alliance")
                        break
                if my_alliance is None:
                    continue
                for t in teams:
                    if t.get("teamNumber") != team_num and t.get("alliance") == my_alliance:
                        partner_num = t.get("teamNumber")
                        break
                if partner_num is None:
                    continue

                # Extract alliance total score — try common field layouts
                alliance_score = None
                if my_alliance == "Red":
                    alliance_score = (match.get("redScore")
                                      or match.get("red_score")
                                      or (match.get("scores") or {}).get("red", {}).get("totalPoints"))
                else:
                    alliance_score = (match.get("blueScore")
                                      or match.get("blue_score")
                                      or (match.get("scores") or {}).get("blue", {}).get("totalPoints"))
                if alliance_score is None:
                    continue

                partner_opr = team_stats.get(partner_num)
                partner_opr_val = partner_opr.opr_total if partner_opr else 0.0
                contributions.append(float(alliance_score) - partner_opr_val)

            if len(contributions) >= min_matches:
                mean = sum(contributions) / len(contributions)
                variance = sum((x - mean) ** 2 for x in contributions) / len(contributions)
                result[team_num] = max(math.sqrt(variance), 5.0)

        return result

    def compute_kalman_opr_for_teams(
        self,
        team_numbers: list[int],
        season: int,
        min_matches: int = 1,
    ) -> dict[int, dict[str, KalmanState]]:
        """Fetch per-match category scores and compute Kalman OPR for all teams.

        For each team this method:
          1. Fetches all season matches via get_team_matches().
          2. Extracts per-category score breakdowns (leave, classified, overflow,
             pattern, depot, base, fouls) from the match scores object.
          3. Identifies the partner team and stores the data for the Kalman pass.
          4. Runs the multi-pass Kalman filter (see kalman_opr.py).

        Penalties are tracked via the 'fouls_committed' category: each team's
        expected foul contribution is modelled separately and applied to the
        opposing alliance during simulation (see simulation.py).

        Args:
            team_numbers: List of team numbers to process.
            season: FTC season year (2025 for DECODE).
            min_matches: Minimum played matches to include a team in the filter.
                         Teams below this threshold start from the prior only.

        Returns:
            {team_num: {category: KalmanState(mean, variance, match_count)}}
        """
        team_processed: dict[int, list[dict]] = {}

        for team_num in team_numbers:
            matches = self.get_team_matches(team_num, season=season)
            records = []

            for match in matches:
                if not match.get("hasBeenPlayed", False):
                    continue

                teams = match.get("teams", [])

                # Identify this team's alliance and partner
                my_alliance = None
                partner_num = None
                for t in teams:
                    if t.get("teamNumber") == team_num:
                        my_alliance = t.get("alliance")
                        break
                if my_alliance is None:
                    continue
                for t in teams:
                    if t.get("teamNumber") != team_num \
                            and t.get("alliance") == my_alliance:
                        partner_num = t.get("teamNumber")
                        break
                if partner_num is None:
                    continue

                # Extract per-category score breakdown
                scores_obj = match.get("scores") or {}
                breakdown = extract_alliance_breakdown(scores_obj, my_alliance)
                if breakdown is None:
                    # Fallback: try to at least get totalPoints so the match
                    # isn't silently dropped; breakdown will be all zeros.
                    breakdown = {cat: 0.0 for cat in gc.CATEGORIES}
                    breakdown.update({
                        "_auto_pts": 0.0, "_endgame_pts": 0.0,
                        "_movement_rp": False, "_goal_rp": False,
                        "_pattern_rp": False, "_pre_foul_total": 0.0,
                        "_total_points": 0.0,
                    })

                records.append({
                    "_breakdown":   breakdown,
                    "_partner_num": partner_num,
                    "_sort_key":    match.get("matchNum", 0),
                })

            if len(records) >= min_matches:
                team_processed[team_num] = records
            else:
                # Not enough data — include with empty records so the team
                # still gets prior-based states in the output dict.
                team_processed[team_num] = []

        prior_mean, prior_var, meas_noise = self._calibrate_kalman_params(
            team_processed
        )
        return compute_kalman_opr(
            team_processed,
            prior_mean=prior_mean,
            prior_var=prior_var,
            measurement_noise=meas_noise,
        )

    def _calibrate_kalman_params(
        self,
        team_processed: dict[int, list[dict]],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Two-pass empirical calibration of Kalman prior means and measurement noise R.

        Pass 1 (rough):
          - Prior mean  = mean(all alliance category scores) / 2   (per-team average)
          - Rough R     = var(all alliance category scores)  / 2   (overestimate; mixes
                          between-team and within-team variance)
          - prior_var is kept large (≥ 10 × rough_R) so the filter adopts the first
            observation quickly — this is critical for teams with only 1-2 matches.

        Pass 2 (residuals):
          - Run a single-pass Kalman with the rough params to get initial OPR estimates.
          - Compute per-match residuals:
                residual = alliance_cat_score − (team_opr_rough + partner_opr_rough)
            These capture only within-team match-to-match variability, NOT between-team
            differences.
          - Final R = var(residuals) per category.  Much smaller than the raw alliance
            variance, leading to sharper OPR estimates and differentiated match
            win-probability predictions.

        Falls back to game_config defaults for any category with fewer than 10 observations.
        """
        from collections import defaultdict

        # ------------------------------------------------------------------ #
        # Pass 1: raw prior means and rough R from alliance observations       #
        # ------------------------------------------------------------------ #
        cat_obs: dict[str, list[float]] = defaultdict(list)
        for records in team_processed.values():
            for record in records:
                bd = record["_breakdown"]
                for cat in gc.CATEGORIES:
                    cat_obs[cat].append(float(bd.get(cat, 0.0)))

        n_obs = len(next(iter(cat_obs.values()), []))
        if n_obs < 10:
            # Not enough data — use game_config defaults unchanged
            return (
                gc.KALMAN_PRIOR_MEAN.copy(),
                gc.KALMAN_PRIOR_VAR.copy(),
                gc.KALMAN_MEASUREMENT_NOISE.copy(),
            )

        rough_prior_mean: dict[str, float] = {}
        rough_R: dict[str, float] = {}
        for cat in gc.CATEGORIES:
            obs = cat_obs[cat]
            mu  = sum(obs) / len(obs)
            var = sum((x - mu) ** 2 for x in obs) / len(obs)
            rough_prior_mean[cat] = mu / 2           # per-team average
            rough_R[cat]          = max(var / 2, 1.0)  # overestimate OK here

        # Keep prior_var large so new observations dominate quickly
        rough_prior_var = {
            cat: max(gc.KALMAN_PRIOR_VAR[cat], 10.0 * rough_R[cat])
            for cat in gc.CATEGORIES
        }

        # ------------------------------------------------------------------ #
        # Rough single-pass Kalman → initial OPR estimates                    #
        # ------------------------------------------------------------------ #
        rough_states = compute_kalman_opr(
            team_processed,
            prior_mean=rough_prior_mean,
            prior_var=rough_prior_var,
            measurement_noise=rough_R,
            num_passes=1,
        )

        # ------------------------------------------------------------------ #
        # Pass 2: compute per-match residuals, derive final R                 #
        # ------------------------------------------------------------------ #
        cat_residuals: dict[str, list[float]] = defaultdict(list)
        for team_num, records in team_processed.items():
            team_s = rough_states.get(team_num, {})
            for record in records:
                partner_num = record["_partner_num"]
                partner_s   = rough_states.get(partner_num, {})
                bd          = record["_breakdown"]
                for cat in gc.CATEGORIES:
                    t_est = team_s.get(cat)
                    p_est = partner_s.get(cat)
                    if t_est is None or p_est is None:
                        continue
                    residual = float(bd.get(cat, 0.0)) - (t_est.mean + p_est.mean)
                    cat_residuals[cat].append(residual)

        final_R: dict[str, float] = {}
        for cat in gc.CATEGORIES:
            res = cat_residuals.get(cat, [])
            if len(res) >= 10:
                mu  = sum(res) / len(res)
                var = sum((x - mu) ** 2 for x in res) / len(res)
                final_R[cat] = max(var, 0.25)
            else:
                final_R[cat] = rough_R[cat]

        # prior_var must stay large relative to final R for fast adaptation
        final_prior_var = {
            cat: max(gc.KALMAN_PRIOR_VAR[cat], 10.0 * final_R[cat])
            for cat in gc.CATEGORIES
        }

        # ------------------------------------------------------------------ #
        # Print calibration summary                                            #
        # ------------------------------------------------------------------ #
        print(f"  Two-pass Kalman calibration from {n_obs} alliance observations:")
        print(f"  {'Category':<20} {'Prior μ':>8} {'R':>8} {'sqrt(R)':>8}")
        print(f"  {'-' * 48}")
        for cat in gc.CATEGORIES:
            print(
                f"  {cat:<20} {rough_prior_mean[cat]:>8.2f}"
                f" {final_R[cat]:>8.2f} {math.sqrt(final_R[cat]):>8.2f}"
            )

        # Propagate calibrated R to gc.KALMAN_MEASUREMENT_NOISE so that
        # TeamStats.cat_std (used by the Monte Carlo simulator) draws on the
        # empirical values rather than the hardcoded game_config defaults.
        gc.KALMAN_MEASUREMENT_NOISE.update(final_R)

        return rough_prior_mean, final_prior_var, final_R

    def get_event_schedule_and_stats(self, season: int, event_code: str):
        """Convenience: fetch matches + team stats for an event.

        Returns (matches, team_stats) where team_stats maps
        team_number -> QuickStats dict.
        """
        matches = self.get_event_matches(season, event_code)
        teams_data = self.get_event_teams(season, event_code)

        # Collect unique team numbers from the event
        team_numbers = set()
        for t in teams_data:
            num = t.get("teamNumber")
            if num:
                team_numbers.add(num)

        stats = self.get_quick_stats_for_teams(sorted(team_numbers), season)
        return matches, stats
