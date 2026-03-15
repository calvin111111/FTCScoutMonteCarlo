"""
FTCScout API client with rate limiting and local caching.

Uses the GraphQL API at api.ftcscout.org/graphql to fetch team statistics,
event data, and match information. Implements request throttling and disk
caching to avoid overwhelming the server.
"""

import json
import os
import time
import hashlib
import requests

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
