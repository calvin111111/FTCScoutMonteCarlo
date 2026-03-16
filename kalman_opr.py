"""
Kalman filter OPR engine for FTC DECODE 2025-2026.

Computes per-category OPR estimates and uncertainties for each team using
a sequential Kalman filter that processes matches chronologically.

The observation model:
    alliance_category_score ≈ team_opr[category] + partner_opr[category] + noise

So each team's contribution is isolated as:
    observation = alliance_score[category] - partner_current_estimate[category]

Multiple passes (default 3) are made over all matches so partner estimates
improve across passes, converging toward a consistent solution.

For each category the filter maintains:
    mean     — best current estimate of the team's OPR for that category
    variance — estimation uncertainty (epistemic); shrinks with more matches

The predictive std dev used in simulation is sqrt(variance + R), combining
estimation uncertainty with inherent match-to-match variability (R).
"""

import math
from dataclasses import dataclass

import game_config as gc


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    """Kalman filter state for (team, category) pair."""
    mean: float
    variance: float
    match_count: int = 0

    @property
    def std_dev(self) -> float:
        """Predictive std dev: estimation uncertainty + measurement noise."""
        R = gc.KALMAN_MEASUREMENT_NOISE.get("_default", 25.0)
        return max(math.sqrt(self.variance + R), 0.5)

    def predictive_std(self, measurement_noise: float) -> float:
        """Predictive std dev for a given measurement noise R."""
        return max(math.sqrt(self.variance + measurement_noise), 0.5)


# ---------------------------------------------------------------------------
# Score breakdown extraction
# ---------------------------------------------------------------------------

def extract_alliance_breakdown(scores_obj: dict, alliance: str) -> dict | None:
    """Parse a match's scores object and return per-category values for one alliance.

    Handles both 'Red'/'Blue' and 'red'/'blue' key conventions.

    Returns a dict with keys matching game_config.CATEGORIES, plus auxiliary
    fields prefixed with '_' used for RP simulation:
        _auto_pts         — total auto points (for TBP1)
        _endgame_pts      — total base/endgame points (for TBP2)
        _movement_rp      — bool: movementRP earned (from API)
        _goal_rp          — bool: goalRP earned (from API)
        _pattern_rp       — bool: patternRP earned (from API)
        _total_points     — totalPoints including fouls from opponent
        _pre_foul_total   — points before opponent fouls are added
    """
    if not scores_obj:
        return None

    # Accept both capitalised and lowercase alliance keys
    alliance_data = (
        scores_obj.get(alliance)
        or scores_obj.get(alliance.lower())
        or scores_obj.get(alliance.capitalize())
    )
    if not alliance_data:
        return None

    def _int(key: int | str, default: int = 0) -> int:
        return int(alliance_data.get(key) or default)

    def _float(key: str, default: float = 0.0) -> float:
        return float(alliance_data.get(key) or default)

    def _bool(key: str) -> bool:
        return bool(alliance_data.get(key, False))

    auto_classified_count  = _int("autoClassifiedArtifacts")
    auto_overflow_count    = _int("autoOverflowArtifacts")
    dc_classified_count    = _int("teleopClassifiedArtifacts")
    dc_overflow_count      = _int("teleopOverflowArtifacts")

    auto_leave_pts      = _float("autoLeavePoints")
    auto_classified_pts = auto_classified_count * gc.CLASSIFIED_ARTIFACT_PTS
    auto_overflow_pts   = auto_overflow_count   * gc.OVERFLOW_ARTIFACT_PTS
    auto_pattern_pts    = _float("autoPatternPoints")
    dc_classified_pts   = dc_classified_count   * gc.CLASSIFIED_ARTIFACT_PTS
    dc_overflow_pts     = dc_overflow_count     * gc.OVERFLOW_ARTIFACT_PTS
    dc_pattern_pts      = _float("teleopPatternPoints")
    dc_depot_pts        = _float("teleopDepotPoints")
    dc_base_pts         = _float("teleopBasePoints")
    fouls_committed     = _float("foulPointsCommitted")

    auto_pts    = auto_leave_pts + auto_classified_pts + auto_overflow_pts + auto_pattern_pts
    endgame_pts = dc_base_pts

    return {
        # ---- Kalman categories ----
        "auto_leave":       auto_leave_pts,
        "auto_classified":  auto_classified_pts,
        "auto_overflow":    auto_overflow_pts,
        "auto_pattern":     auto_pattern_pts,
        "dc_classified":    dc_classified_pts,
        "dc_overflow":      dc_overflow_pts,
        "dc_pattern":       dc_pattern_pts,
        "dc_depot":         dc_depot_pts,
        "dc_base":          dc_base_pts,
        "fouls_committed":  fouls_committed,
        # ---- Auxiliary (not Kalman categories) ----
        "_auto_pts":        auto_pts,
        "_endgame_pts":     endgame_pts,
        "_movement_rp":     _bool("movementRP"),
        "_goal_rp":         _bool("goalRP"),
        "_pattern_rp":      _bool("patternRP"),
        "_pre_foul_total":  _float("preFoulTotal"),
        "_total_points":    _float("totalPoints"),
    }


# ---------------------------------------------------------------------------
# Kalman filter update step
# ---------------------------------------------------------------------------

def _kalman_update(state: KalmanState, observation: float,
                   Q: float, R: float) -> KalmanState:
    """One Kalman predict-then-update step.

    Args:
        state: current (mean, variance) estimate.
        observation: this team's estimated contribution for one match.
        Q: process noise — how much the true OPR drifts per match.
        R: measurement noise — variance of each per-match observation.

    Returns:
        Updated KalmanState.
    """
    # Predict: true OPR may have drifted
    pred_variance = state.variance + Q

    # Innovation
    innovation = observation - state.mean
    innov_var   = pred_variance + R

    # Kalman gain
    K = pred_variance / innov_var

    # Update
    new_mean     = state.mean + K * innovation
    new_variance = (1.0 - K) * pred_variance

    return KalmanState(
        mean=new_mean,
        variance=new_variance,
        match_count=state.match_count + 1,
    )


# ---------------------------------------------------------------------------
# Multi-pass Kalman OPR computation
# ---------------------------------------------------------------------------

def compute_kalman_opr(
    team_processed_matches: dict[int, list[dict]],
    categories: list[str] | None = None,
    num_passes: int | None = None,
    prior_mean: dict[str, float] | None = None,
    prior_var:  dict[str, float] | None = None,
    process_noise: dict[str, float] | None = None,
    measurement_noise: dict[str, float] | None = None,
) -> dict[int, dict[str, KalmanState]]:
    """Compute per-category Kalman OPR for every team.

    Args:
        team_processed_matches:
            {team_num: [match_record, ...]}
            Each match_record must have:
                "_breakdown":   dict from extract_alliance_breakdown()
                "_partner_num": int (partner team number)
                "_sort_key":    int (match number; used for chronological order)

        categories: list of category names to estimate (default: gc.CATEGORIES).
        num_passes: convergence passes (default: gc.KALMAN_PASSES).
        prior_mean / prior_var / process_noise / measurement_noise:
            Override game_config defaults when provided.

    Returns:
        {team_num: {category: KalmanState}}
    """
    cats   = categories     or gc.CATEGORIES
    passes = num_passes     or gc.KALMAN_PASSES
    p_mean = prior_mean     or gc.KALMAN_PRIOR_MEAN
    p_var  = prior_var      or gc.KALMAN_PRIOR_VAR
    Q      = process_noise  or gc.KALMAN_PROCESS_NOISE
    R      = measurement_noise or gc.KALMAN_MEASUREMENT_NOISE

    all_teams = set(team_processed_matches.keys())

    def _fresh_states() -> dict[int, dict[str, KalmanState]]:
        return {
            team: {cat: KalmanState(p_mean[cat], p_var[cat]) for cat in cats}
            for team in all_teams
        }

    # States used by partners in the current pass
    states = _fresh_states()

    for _ in range(passes):
        new_states = _fresh_states()

        for team_num, match_records in team_processed_matches.items():
            # Sort chronologically (re-sort each pass — records are immutable)
            for record in sorted(match_records, key=lambda m: m["_sort_key"]):
                breakdown   = record["_breakdown"]
                partner_num = record["_partner_num"]

                # Use previous-pass partner states as the partner estimate
                partner_states = states.get(partner_num, {})

                for cat in cats:
                    alliance_cat_score = breakdown.get(cat, 0.0)
                    partner_est = partner_states.get(
                        cat, KalmanState(p_mean[cat], p_var[cat])
                    ).mean

                    # Observation: this team's isolated contribution
                    observation = alliance_cat_score - partner_est

                    current = new_states[team_num][cat]
                    new_states[team_num][cat] = _kalman_update(
                        current, observation, Q[cat], R[cat]
                    )

        states = new_states

    return states
