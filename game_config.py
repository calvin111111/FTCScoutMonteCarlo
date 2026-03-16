"""
FTC DECODE 2025-2026 game scoring constants and ranking point thresholds.

Point values verified from DecodeDescriptor.ts (github.com/ftc-scout/ftc-scout).
RP thresholds derived from DECODE Competition Manual Table 10-3.

!! IMPORTANT: Verify threshold values against your event type before running. !!
  - Qualifying tournaments / league meets  → use QUALIFIER thresholds
  - Regional championships (RCMP)          → use REGIONAL thresholds

Regional championship thresholds (21 / 42 / 22) are sourced from FTC community
reporting; qualifying thresholds are estimated from scoring data and should be
confirmed against Section 10.5.4 Table 10-3 of the official game manual at
  https://ftc-resources.firstinspires.org/ftc/game/manual
"""

SEASON = 2025

# ---------------------------------------------------------------------------
# Point values per scoring action (from DecodeDescriptor.ts)
# ---------------------------------------------------------------------------

LEAVE_PTS_PER_ROBOT = 3             # each robot that crosses the launch line in auto

CLASSIFIED_ARTIFACT_PTS = 3        # per classified artifact (auto or teleop)
OVERFLOW_ARTIFACT_PTS = 1          # per overflow artifact (auto or teleop)
DEPOT_ARTIFACT_PTS = 1             # per artifact scored in the depot (teleop)

PARTIAL_BASE_RETURN_PTS = 5        # per robot with partial base return (endgame)
FULL_BASE_RETURN_PTS = 10          # per robot with full base return (endgame)
BOTH_FULL_BASE_BONUS_PTS = 10      # bonus if BOTH robots return fully

MAJOR_FOUL_PTS = 15                # awarded to opposing alliance per major foul
MINOR_FOUL_PTS = 5                 # awarded to opposing alliance per minor foul

# ---------------------------------------------------------------------------
# Scoring categories tracked by the Kalman filter
# ---------------------------------------------------------------------------
# Each category corresponds to one field (or derived field) from
# AllianceScores2025TradFtcApi in the FTCScout API.

CATEGORIES = [
    "auto_leave",       # autoLeavePoints  (2 robots × 3 pts max)
    "auto_classified",  # autoClassifiedArtifacts × CLASSIFIED_ARTIFACT_PTS
    "auto_overflow",    # autoOverflowArtifacts × OVERFLOW_ARTIFACT_PTS
    "auto_pattern",     # autoPatternPoints
    "dc_classified",    # teleopClassifiedArtifacts × CLASSIFIED_ARTIFACT_PTS
    "dc_overflow",      # teleopOverflowArtifacts × OVERFLOW_ARTIFACT_PTS
    "dc_pattern",       # teleopPatternPoints
    "dc_depot",         # teleopDepotPoints
    "dc_base",          # teleopBasePoints  (robot returns to base)
    "foul_pts_received", # foulPointsCommitted by the OPPONENT — awarded TO
                         # this alliance and included in the reported score
]

# All categories contribute additively to an alliance's own score
POSITIVE_CATEGORIES = CATEGORIES

# ---------------------------------------------------------------------------
# Ranking Point (RP) thresholds — Table 10-3
# ---------------------------------------------------------------------------
# Each qualification match awards up to 4 RP per alliance:
#   1 RP for win (base), or 0 for loss, 0.5 each for tie  ← standard
#   + movementRP bonus  (up to +1)
#   + goalRP bonus      (up to +1)
#   + patternRP bonus   (up to +1)
#
# movementRP — awarded when both robots leave the launch line in AUTO.
#   Threshold: autoLeavePoints >= 2 × LEAVE_PTS_PER_ROBOT  (= 6 pts always)
#   (This is the same at all event tiers — it's a binary "both left" check.)
MOVEMENT_RP_THRESHOLD = 2 * LEAVE_PTS_PER_ROBOT   # 6

# goalRP — awarded when the alliance classifies enough artifacts.
#   Threshold is on TOTAL classified artifact POINTS (auto + teleop).
#   Equivalent classified count = threshold / CLASSIFIED_ARTIFACT_PTS.
#
#   Table 10-3  "All Other Events" (qualifiers, league meets):
GOAL_RP_CLASSIFIED_PTS_QUALIFIER = 30    # ≈ 10 classified artifacts; verify from manual
#   Table 10-3  Regional Championship (RCMP) — from reported "42" value:
GOAL_RP_CLASSIFIED_PTS_REGIONAL = 42    # = 14 classified artifacts

# patternRP — awarded when the alliance scores enough pattern points.
#   Threshold is on TOTAL pattern POINTS (auto + teleop combined).
#
#   Table 10-3  "All Other Events":
PATTERN_RP_PTS_QUALIFIER = 14           # verify from manual
#   Table 10-3  Regional Championship — from reported "22" value:
PATTERN_RP_PTS_REGIONAL = 22

# ---------------------------------------------------------------------------
# Kalman filter hyperparameters
# ---------------------------------------------------------------------------
# These control how the Kalman filter weights prior beliefs vs new observations.
# Tune these based on observed convergence / accuracy for your dataset.

# Prior: starting belief before any match data. Mean = 0, Variance = 50² (very uncertain)
KALMAN_PRIOR_MEAN: dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
KALMAN_PRIOR_VAR: dict[str, float] = {cat: 50.0 ** 2 for cat in CATEGORIES}

# Process noise Q: how much a team's "true" strength drifts per match.
# Small Q → filter assumes stable performance; larger Q → more responsive to recent data.
KALMAN_PROCESS_NOISE: dict[str, float] = {cat: 0.5 for cat in CATEGORIES}

# Measurement noise R: variance in each observation due to partner estimation error
# and inherent match-to-match variability. Larger R → slower convergence.
KALMAN_MEASUREMENT_NOISE: dict[str, float] = {
    "auto_leave":       4.0,    # max is small (6 pts), little variance
    "auto_classified":  25.0,
    "auto_overflow":    4.0,
    "auto_pattern":     16.0,
    "dc_classified":    64.0,   # high variance in scoring efficiency
    "dc_overflow":      16.0,
    "dc_pattern":       36.0,
    "dc_depot":         16.0,
    "dc_base":          36.0,   # some variance in endgame reliability
    "foul_pts_received": 25.0,
}

# Number of Kalman convergence passes (3 is typically sufficient)
KALMAN_PASSES = 3
