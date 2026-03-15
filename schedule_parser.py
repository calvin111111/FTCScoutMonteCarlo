"""
Parse match schedules from FTCScout API responses and CSV files into
ScheduledMatch objects for simulation.
"""

import csv
import io
from simulation import ScheduledMatch


def parse_ftcscout_matches(matches_data: list[dict]) -> list[ScheduledMatch]:
    """Parse matches from the FTCScout REST/GraphQL API response.

    Each match entry has teams as a list of TeamMatchParticipation objects
    with fields: teamNumber, alliance ("Red"/"Blue"), station (0/1).

    Only includes matches that have not been played yet (hasBeenPlayed=False).
    If all matches are already played, includes all of them so results can
    be compared against actuals.
    """
    unplayed = [m for m in matches_data if not m.get("hasBeenPlayed", False)]
    target = unplayed if unplayed else matches_data

    schedule = []
    for m in target:
        match_id = m.get("id", 0)
        level = m.get("tournamentLevel", "QUALS")
        num = m.get("matchNum", match_id)
        series = m.get("series", 0)

        if level == "QUALS":
            name = f"Q-{num}"
        elif "SEMI" in str(level).upper():
            name = f"SF-{series}-{num}"
        elif "FINAL" in str(level).upper():
            name = f"F-{num}"
        else:
            name = f"{level}-{num}"

        teams = m.get("teams", [])
        red = [t["teamNumber"] for t in teams
               if t.get("alliance") == "Red"]
        blue = [t["teamNumber"] for t in teams
                if t.get("alliance") == "Blue"]

        if not red or not blue:
            continue

        schedule.append(ScheduledMatch(
            match_id=match_id,
            match_name=name,
            red_teams=red,
            blue_teams=blue,
        ))

    return schedule


def parse_csv_schedule(csv_text: str) -> list[ScheduledMatch]:
    """Parse a match schedule from CSV.

    Expected columns (header required):
        match_id, match_name, red1, red2, blue1, blue2

    Extra red/blue columns (red3, blue3) are supported for larger alliances.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    schedule = []
    for row in reader:
        match_id = int(row.get("match_id", 0))
        match_name = row.get("match_name", f"M-{match_id}")

        red = []
        blue = []
        for key in sorted(row.keys()):
            val = row[key].strip() if row[key] else ""
            if not val:
                continue
            if key.startswith("red") and key != "red_teams":
                red.append(int(val))
            elif key.startswith("blue") and key != "blue_teams":
                blue.append(int(val))

        if not red or not blue:
            continue

        schedule.append(ScheduledMatch(
            match_id=match_id,
            match_name=match_name,
            red_teams=red,
            blue_teams=blue,
        ))

    return schedule


def parse_csv_file(path: str) -> list[ScheduledMatch]:
    """Parse a match schedule from a CSV file on disk."""
    with open(path, "r") as f:
        return parse_csv_schedule(f.read())
