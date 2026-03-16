"""
Monte Carlo simulation engine for FTC DECODE 2025-2026 match prediction.

Given per-category Kalman OPR statistics and a match schedule, simulates many
possible outcomes to estimate:
  - Win/loss/tie probabilities for each match
  - Expected scores (penalties included as additive foul_pts_received, matching
    how scores are reported in FTC matches)
  - Final ranking distributions including bonus RP (movementRP, goalRP, patternRP)

Each category is sampled from N(kalman_mean, predictive_std) per team. Alliance
scores are the sum of each team's category samples, including foul_pts_received
(the foul points awarded to this alliance from opponent infractions).

Bonus RP logic (DECODE 2025-2026):
  - movementRP: alliance auto_leave_pts  >= MOVEMENT_RP_THRESHOLD
  - goalRP:     alliance total classified_pts / 3 >= GOAL_RP_CLASSIFIED_THRESHOLD
  - patternRP:  alliance total pattern_pts >= PATTERN_RP_PTS_THRESHOLD

Thresholds are configurable — see game_config.py.
"""

import math
import numpy as np
from dataclasses import dataclass, field

import game_config as gc
from kalman_opr import KalmanState


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class TeamStats:
    """Per-category Kalman OPR statistics for one team."""
    number: int
    match_count: int = 0

    # {category: KalmanState}  — populated from Kalman filter output
    kalman: dict[str, KalmanState] = field(default_factory=dict)

    # ---- Convenience accessors ----------------------------------------

    def cat_mean(self, cat: str) -> float:
        """Expected score for one category (floored at 0)."""
        state = self.kalman.get(cat)
        return max(state.mean if state else 0.0, 0.0)

    def cat_std(self, cat: str, measurement_noise: float | None = None) -> float:
        """Predictive std dev for one category."""
        state = self.kalman.get(cat)
        if state is None:
            return max(gc.KALMAN_MEASUREMENT_NOISE.get(cat, 25.0) ** 0.5, 0.5)
        R = measurement_noise if measurement_noise is not None \
            else gc.KALMAN_MEASUREMENT_NOISE.get(cat, 25.0)
        return state.predictive_std(R)

    # ---- Derived total OPR (for display / backward compat) ------------

    @property
    def opr_total(self) -> float:
        """Total expected score per match, including expected foul points received."""
        return max(sum(self.cat_mean(c) for c in gc.POSITIVE_CATEGORIES), 0.0)

    @property
    def opr_auto(self) -> float:
        auto_cats = ["auto_leave", "auto_classified", "auto_overflow", "auto_pattern"]
        return max(sum(self.cat_mean(c) for c in auto_cats), 0.0)

    @property
    def opr_teleop(self) -> float:
        dc_cats = ["dc_classified", "dc_overflow", "dc_pattern", "dc_depot"]
        return max(sum(self.cat_mean(c) for c in dc_cats), 0.0)

    @property
    def opr_endgame(self) -> float:
        return self.cat_mean("dc_base")

    @property
    def std_dev(self) -> float:
        """Combined predictive std dev for total score."""
        total_var = sum(
            self.cat_std(c) ** 2 for c in gc.POSITIVE_CATEGORIES
        )
        return max(math.sqrt(total_var), 3.0)


@dataclass
class ScheduledMatch:
    """A single match from the schedule."""
    match_id: int
    match_name: str      # e.g. "Q-1", "SF-1-1"
    red_teams: list[int]
    blue_teams: list[int]


@dataclass
class MatchResult:
    """Aggregated simulation results for one match."""
    match: ScheduledMatch
    red_win_pct: float  = 0.0
    blue_win_pct: float = 0.0
    tie_pct: float      = 0.0
    avg_red_score: float  = 0.0
    avg_blue_score: float = 0.0
    std_red_score: float  = 0.0
    std_blue_score: float = 0.0


@dataclass
class TeamRankingDist:
    """Ranking distribution for one team after simulation."""
    number: int
    avg_rp: float    = 0.0
    avg_tbp: float   = 0.0   # TBP1 = cumulative auto pts
    avg_tbp2: float  = 0.0   # TBP2 = cumulative endgame pts
    avg_rank: float  = 0.0
    rank_counts: dict[int, int] = field(default_factory=dict)
    win_count: float  = 0.0
    loss_count: float = 0.0
    tie_count: float  = 0.0


# -----------------------------------------------------------------------
# Helper: build per-match, per-alliance moment arrays
# -----------------------------------------------------------------------

def _alliance_moments(
    team_stats: dict[int, TeamStats],
    teams: list[int],
    categories: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return {cat: mean} and {cat: std} summed across an alliance's teams."""
    means = {}
    stds  = {}
    for cat in categories:
        mu  = sum(team_stats[t].cat_mean(cat) for t in teams)
        var = sum(team_stats[t].cat_std(cat) ** 2 for t in teams)
        means[cat] = mu
        stds[cat]  = math.sqrt(var)
    return means, stds


# -----------------------------------------------------------------------
# Simulation engine
# -----------------------------------------------------------------------

class MonteCarloSimulator:
    """Run Monte Carlo simulations over an FTC DECODE event schedule."""

    def __init__(
        self,
        team_stats: dict[int, TeamStats],
        schedule: list[ScheduledMatch],
        num_simulations: int = 10_000,
        seed: int | None = None,
        goal_rp_threshold: float   = gc.GOAL_RP_CLASSIFIED_PTS_QUALIFIER,
        pattern_rp_threshold: float = gc.PATTERN_RP_PTS_QUALIFIER,
        movement_rp_threshold: float = gc.MOVEMENT_RP_THRESHOLD,
    ):
        """
        Args:
            team_stats: Mapping of team number -> TeamStats (Kalman OPR).
            schedule: List of scheduled matches.
            num_simulations: How many full-event simulations to run.
            seed: Random seed for reproducibility.
            goal_rp_threshold: Total classified artifact POINTS threshold for goalRP.
            pattern_rp_threshold: Total pattern POINTS threshold for patternRP.
            movement_rp_threshold: Auto leave POINTS threshold for movementRP.
        """
        self.team_stats          = team_stats
        self.schedule            = schedule
        self.num_simulations     = num_simulations
        self.rng                 = np.random.default_rng(seed)
        self.goal_rp_threshold   = goal_rp_threshold
        self.pattern_rp_threshold = pattern_rp_threshold
        self.movement_rp_threshold = movement_rp_threshold

        # Collect all team numbers from the schedule
        self.all_teams: set[int] = set()
        for m in schedule:
            self.all_teams.update(m.red_teams)
            self.all_teams.update(m.blue_teams)

        # Fill in default (prior-only) stats for any team missing data
        for t in self.all_teams:
            if t not in self.team_stats:
                self.team_stats[t] = TeamStats(number=t)

    # ------------------------------------------------------------------
    # Score sampling helpers
    # ------------------------------------------------------------------

    def _sample(self, mean: float, std: float) -> np.ndarray:
        """Sample num_simulations values from N(mean, std).

        No floor is applied here — negative values are valid OPR statistical
        artifacts and are handled at comparison/display time.  This avoids an
        artificial pile-up of zeros that would inflate tie percentages.
        """
        return self.rng.normal(mean, max(std, 0.1), self.num_simulations)

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[MatchResult], dict[int, TeamRankingDist]]:
        """Run the full Monte Carlo simulation.

        Returns:
            (match_results, ranking_distributions)
        """
        n   = self.num_simulations
        M   = len(self.schedule)
        cats = gc.CATEGORIES

        # Pre-allocate arrays: shape (n, M)
        # Total scores include foul_pts_received (already in opr_total)
        red_base  = np.zeros((n, M))
        blue_base = np.zeros((n, M))

        # RP-threshold category arrays
        red_auto_leave  = np.zeros((n, M))   # for movementRP
        blue_auto_leave = np.zeros((n, M))

        red_classified  = np.zeros((n, M))   # auto_classified + dc_classified pts
        blue_classified = np.zeros((n, M))   # for goalRP

        red_pattern  = np.zeros((n, M))      # auto_pattern + dc_pattern pts
        blue_pattern = np.zeros((n, M))      # for patternRP

        # Auto pts = leave + classified + overflow + pattern  (for TBP1)
        red_auto  = np.zeros((n, M))
        blue_auto = np.zeros((n, M))

        # Endgame/base pts (for TBP2)
        red_endgame  = np.zeros((n, M))
        blue_endgame = np.zeros((n, M))

        for j, match in enumerate(self.schedule):
            R = match.red_teams
            B = match.blue_teams

            # ------ Total non-foul score -----------------------------------
            # Sum means and combine variances across partners
            red_total_mean  = sum(self.team_stats[t].opr_total for t in R)
            red_total_std   = math.sqrt(sum(self.team_stats[t].std_dev**2 for t in R))
            blue_total_mean = sum(self.team_stats[t].opr_total for t in B)
            blue_total_std  = math.sqrt(sum(self.team_stats[t].std_dev**2 for t in B))

            red_base[:, j]  = self._sample(red_total_mean,  red_total_std)
            blue_base[:, j] = self._sample(blue_total_mean, blue_total_std)

            # ------ RP threshold categories --------------------------------
            # auto_leave
            r_leave_mean = sum(self.team_stats[t].cat_mean("auto_leave") for t in R)
            r_leave_std  = math.sqrt(sum(self.team_stats[t].cat_std("auto_leave")**2 for t in R))
            b_leave_mean = sum(self.team_stats[t].cat_mean("auto_leave") for t in B)
            b_leave_std  = math.sqrt(sum(self.team_stats[t].cat_std("auto_leave")**2 for t in B))
            red_auto_leave[:, j]  = self._sample(r_leave_mean, r_leave_std)
            blue_auto_leave[:, j] = self._sample(b_leave_mean, b_leave_std)

            # classified pts (auto + dc combined)
            r_class_mean = sum(
                self.team_stats[t].cat_mean("auto_classified")
                + self.team_stats[t].cat_mean("dc_classified")
                for t in R
            )
            r_class_std = math.sqrt(sum(
                self.team_stats[t].cat_std("auto_classified")**2
                + self.team_stats[t].cat_std("dc_classified")**2
                for t in R
            ))
            b_class_mean = sum(
                self.team_stats[t].cat_mean("auto_classified")
                + self.team_stats[t].cat_mean("dc_classified")
                for t in B
            )
            b_class_std = math.sqrt(sum(
                self.team_stats[t].cat_std("auto_classified")**2
                + self.team_stats[t].cat_std("dc_classified")**2
                for t in B
            ))
            red_classified[:, j]  = self._sample(r_class_mean, r_class_std)
            blue_classified[:, j] = self._sample(b_class_mean, b_class_std)

            # pattern pts (auto + dc combined)
            r_pat_mean = sum(
                self.team_stats[t].cat_mean("auto_pattern")
                + self.team_stats[t].cat_mean("dc_pattern")
                for t in R
            )
            r_pat_std = math.sqrt(sum(
                self.team_stats[t].cat_std("auto_pattern")**2
                + self.team_stats[t].cat_std("dc_pattern")**2
                for t in R
            ))
            b_pat_mean = sum(
                self.team_stats[t].cat_mean("auto_pattern")
                + self.team_stats[t].cat_mean("dc_pattern")
                for t in B
            )
            b_pat_std = math.sqrt(sum(
                self.team_stats[t].cat_std("auto_pattern")**2
                + self.team_stats[t].cat_std("dc_pattern")**2
                for t in B
            ))
            red_pattern[:, j]  = self._sample(r_pat_mean, r_pat_std)
            blue_pattern[:, j] = self._sample(b_pat_mean, b_pat_std)

            # ------ Auto pts for TBP1 ------------------------------------
            auto_cats = ["auto_leave", "auto_classified", "auto_overflow", "auto_pattern"]
            r_auto_mean = sum(self.team_stats[t].cat_mean(c) for t in R for c in auto_cats)
            r_auto_std  = math.sqrt(sum(self.team_stats[t].cat_std(c)**2 for t in R for c in auto_cats))
            b_auto_mean = sum(self.team_stats[t].cat_mean(c) for t in B for c in auto_cats)
            b_auto_std  = math.sqrt(sum(self.team_stats[t].cat_std(c)**2 for t in B for c in auto_cats))
            red_auto[:, j]  = self._sample(r_auto_mean, r_auto_std)
            blue_auto[:, j] = self._sample(b_auto_mean, b_auto_std)

            # ------ Endgame pts for TBP2 ---------------------------------
            r_eg_mean = sum(self.team_stats[t].cat_mean("dc_base") for t in R)
            r_eg_std  = math.sqrt(sum(self.team_stats[t].cat_std("dc_base")**2 for t in R))
            b_eg_mean = sum(self.team_stats[t].cat_mean("dc_base") for t in B)
            b_eg_std  = math.sqrt(sum(self.team_stats[t].cat_std("dc_base")**2 for t in B))
            red_endgame[:, j]  = self._sample(r_eg_mean, r_eg_std)
            blue_endgame[:, j] = self._sample(b_eg_mean, b_eg_std)

        # ------------------------------------------------------------------
        # Final scores: foul_pts_received is already included in opr_total/base.
        # Round to integers for win/loss/tie comparison — FTC scores are integer
        # points and rounding eliminates the artificial tie spike that arises from
        # comparing continuous floats (especially the 0-vs-0 pile at low OPR).
        # ------------------------------------------------------------------
        red_scores  = red_base
        blue_scores = blue_base
        red_scores_int  = np.round(red_base).astype(int)
        blue_scores_int = np.round(blue_base).astype(int)

        # ------------------------------------------------------------------
        # Bonus RP masks   shape: (n, M)  boolean
        # ------------------------------------------------------------------
        red_movement_rp  = red_auto_leave  >= self.movement_rp_threshold
        blue_movement_rp = blue_auto_leave >= self.movement_rp_threshold

        # classified_pts / CLASSIFIED_ARTIFACT_PTS = classified artifact count
        red_goal_rp  = (red_classified  / gc.CLASSIFIED_ARTIFACT_PTS) >= self.goal_rp_threshold
        blue_goal_rp = (blue_classified / gc.CLASSIFIED_ARTIFACT_PTS) >= self.goal_rp_threshold

        red_pattern_rp  = red_pattern  >= self.pattern_rp_threshold
        blue_pattern_rp = blue_pattern >= self.pattern_rp_threshold

        # ------------------------------------------------------------------
        # Match-level results
        # ------------------------------------------------------------------
        match_results: list[MatchResult] = []
        for j, match in enumerate(self.schedule):
            ri = red_scores_int[:, j]
            bi = blue_scores_int[:, j]
            red_wins  = int(np.sum(ri > bi))
            blue_wins = int(np.sum(bi > ri))
            ties      = n - red_wins - blue_wins

            # Display: clip continuous scores at 0 (scores can't be negative in FTC)
            r_disp = np.maximum(red_scores[:, j],  0.0)
            b_disp = np.maximum(blue_scores[:, j], 0.0)
            match_results.append(MatchResult(
                match=match,
                red_win_pct   = red_wins  / n * 100,
                blue_win_pct  = blue_wins / n * 100,
                tie_pct       = ties      / n * 100,
                avg_red_score = float(np.mean(r_disp)),
                avg_blue_score= float(np.mean(b_disp)),
                std_red_score = float(np.std(r_disp)),
                std_blue_score= float(np.std(b_disp)),
            ))

        # ------------------------------------------------------------------
        # Ranking distributions
        # ------------------------------------------------------------------
        ranking_dists = {t: TeamRankingDist(number=t) for t in self.all_teams}
        sorted_teams  = sorted(self.all_teams)
        team_idx      = {t: i for i, t in enumerate(sorted_teams)}

        team_rp   = np.zeros((n, len(self.all_teams)))
        team_tbp1 = np.zeros((n, len(self.all_teams)))  # cumulative auto pts
        team_tbp2 = np.zeros((n, len(self.all_teams)))  # cumulative endgame pts

        for j, match in enumerate(self.schedule):
            ri  = red_scores_int[:, j]
            bi  = blue_scores_int[:, j]
            rwm = ri > bi      # red win mask
            bwm = bi > ri      # blue win mask
            tm  = ~rwm & ~bwm  # tie mask

            r_mrp = red_movement_rp[:, j].astype(float)
            r_grp = red_goal_rp[:, j].astype(float)
            r_prp = red_pattern_rp[:, j].astype(float)
            b_mrp = blue_movement_rp[:, j].astype(float)
            b_grp = blue_goal_rp[:, j].astype(float)
            b_prp = blue_pattern_rp[:, j].astype(float)

            for t in match.red_teams:
                idx = team_idx[t]
                team_rp[:, idx] += (
                    rwm.astype(float) * 2
                    + tm.astype(float)
                    + r_mrp + r_grp + r_prp
                )
                team_tbp1[:, idx] += np.maximum(red_auto[:, j],  0.0)
                team_tbp2[:, idx] += np.maximum(red_endgame[:, j], 0.0)

            for t in match.blue_teams:
                idx = team_idx[t]
                team_rp[:, idx] += (
                    bwm.astype(float) * 2
                    + tm.astype(float)
                    + b_mrp + b_grp + b_prp
                )
                team_tbp1[:, idx] += np.maximum(blue_auto[:, j],  0.0)
                team_tbp2[:, idx] += np.maximum(blue_endgame[:, j], 0.0)

        # Compute ranks per simulation: primary RP, secondary TBP1, tertiary TBP2
        for sim_i in range(n):
            rp  = team_rp[sim_i]
            tb1 = team_tbp1[sim_i]
            tb2 = team_tbp2[sim_i]
            order = np.lexsort((-tb2, -tb1, -rp))
            for rank_0, team_i in enumerate(order):
                t    = sorted_teams[team_i]
                rank = rank_0 + 1
                dist = ranking_dists[t]
                dist.rank_counts[rank] = dist.rank_counts.get(rank, 0) + 1

        # Average statistics
        for t in sorted_teams:
            idx  = team_idx[t]
            dist = ranking_dists[t]
            dist.avg_rp   = float(np.mean(team_rp[:, idx]))
            dist.avg_tbp  = float(np.mean(team_tbp1[:, idx]))
            dist.avg_tbp2 = float(np.mean(team_tbp2[:, idx]))
            total_rank    = sum(r * c for r, c in dist.rank_counts.items())
            dist.avg_rank = total_rank / n

        # Win/loss/tie per team
        for j, match in enumerate(self.schedule):
            ri = red_scores_int[:, j]
            bi = blue_scores_int[:, j]
            rw = int(np.sum(ri > bi))
            bw = int(np.sum(bi > ri))
            ti = n - rw - bw
            for t in match.red_teams:
                ranking_dists[t].win_count  += rw / n
                ranking_dists[t].loss_count += bw / n
                ranking_dists[t].tie_count  += ti / n
            for t in match.blue_teams:
                ranking_dists[t].win_count  += bw / n
                ranking_dists[t].loss_count += rw / n
                ranking_dists[t].tie_count  += ti / n

        return match_results, ranking_dists
