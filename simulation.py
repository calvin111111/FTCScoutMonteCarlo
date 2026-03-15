"""
Monte Carlo simulation engine for FTC match prediction.

Given team OPR statistics (total, auto, driver-controlled, endgame) and a
match schedule, simulates many possible outcomes to estimate:
  - Win/loss/tie probabilities for each match
  - Expected scores
  - Final ranking distributions

Each team's per-match score is modeled as a normal distribution centered on
their OPR with a standard deviation derived from observed FTC score variance.
Alliance scores are the sum of the two partner teams' individual scores.
"""

import numpy as np
from dataclasses import dataclass, field


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class TeamStats:
    """OPR-based statistics for one team."""
    number: int
    opr_total: float = 0.0
    opr_auto: float = 0.0
    opr_teleop: float = 0.0
    opr_endgame: float = 0.0
    match_count: int = 0

    @property
    def std_dev(self) -> float:
        """Estimate per-match score standard deviation.

        FTC scores typically have a coefficient of variation around 15-25%.
        We use 20% of the OPR as a reasonable default. Teams with very few
        matches get a wider distribution to express greater uncertainty.
        """
        base_cv = 0.20
        if self.match_count < 5:
            base_cv = 0.30
        return max(self.opr_total * base_cv, 5.0)


@dataclass
class ScheduledMatch:
    """A single match from the schedule."""
    match_id: int
    match_name: str  # e.g. "Q-1", "SF-1-1"
    red_teams: list[int]  # team numbers
    blue_teams: list[int]


@dataclass
class MatchResult:
    """Aggregated simulation results for one match."""
    match: ScheduledMatch
    red_win_pct: float = 0.0
    blue_win_pct: float = 0.0
    tie_pct: float = 0.0
    avg_red_score: float = 0.0
    avg_blue_score: float = 0.0
    std_red_score: float = 0.0
    std_blue_score: float = 0.0


@dataclass
class TeamRankingDist:
    """Ranking distribution for one team after simulation."""
    number: int
    avg_rp: float = 0.0
    avg_tbp: float = 0.0  # tie-breaker points (total score)
    avg_rank: float = 0.0
    rank_counts: dict[int, int] = field(default_factory=dict)
    win_count: float = 0.0
    loss_count: float = 0.0
    tie_count: float = 0.0


# -----------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------

class MonteCarloSimulator:
    """Run Monte Carlo simulations over an FTC event schedule."""

    def __init__(self, team_stats: dict[int, TeamStats],
                 schedule: list[ScheduledMatch],
                 num_simulations: int = 10000,
                 seed: int | None = None):
        """
        Args:
            team_stats: Mapping of team number -> TeamStats.
            schedule: List of scheduled matches.
            num_simulations: How many full-event simulations to run.
            seed: Random seed for reproducibility.
        """
        self.team_stats = team_stats
        self.schedule = schedule
        self.num_simulations = num_simulations
        self.rng = np.random.default_rng(seed)

        # Collect all team numbers from the schedule
        self.all_teams = set()
        for m in schedule:
            self.all_teams.update(m.red_teams)
            self.all_teams.update(m.blue_teams)

        # Fill in default stats for any team missing data
        for t in self.all_teams:
            if t not in self.team_stats:
                self.team_stats[t] = TeamStats(number=t)

    def _simulate_alliance_score(self, team_numbers: list[int]) -> float:
        """Sample a single alliance score from the team distributions."""
        total = 0.0
        for num in team_numbers:
            stats = self.team_stats[num]
            score = self.rng.normal(stats.opr_total, stats.std_dev)
            total += max(score, 0.0)  # scores can't be negative
        return total

    def _simulate_match(self, match: ScheduledMatch) -> tuple[float, float]:
        """Simulate one match, returning (red_score, blue_score)."""
        red = self._simulate_alliance_score(match.red_teams)
        blue = self._simulate_alliance_score(match.blue_teams)
        return round(red, 1), round(blue, 1)

    def run(self) -> tuple[list[MatchResult], dict[int, TeamRankingDist]]:
        """Run the full Monte Carlo simulation.

        Returns:
            (match_results, ranking_distributions)
        """
        num_matches = len(self.schedule)

        # Pre-allocate arrays: shape (num_simulations, num_matches)
        red_scores = np.zeros((self.num_simulations, num_matches))
        blue_scores = np.zeros((self.num_simulations, num_matches))

        # Vectorised score generation for speed
        for j, match in enumerate(self.schedule):
            red_mean = sum(self.team_stats[t].opr_total
                           for t in match.red_teams)
            red_std = np.sqrt(sum(self.team_stats[t].std_dev ** 2
                                  for t in match.red_teams))
            blue_mean = sum(self.team_stats[t].opr_total
                            for t in match.blue_teams)
            blue_std = np.sqrt(sum(self.team_stats[t].std_dev ** 2
                                   for t in match.blue_teams))

            red_scores[:, j] = np.maximum(
                self.rng.normal(red_mean, red_std, self.num_simulations), 0)
            blue_scores[:, j] = np.maximum(
                self.rng.normal(blue_mean, blue_std, self.num_simulations), 0)

        # ------------------------------------------------------------------
        # Aggregate match-level results
        # ------------------------------------------------------------------
        match_results = []
        for j, match in enumerate(self.schedule):
            r = red_scores[:, j]
            b = blue_scores[:, j]
            red_wins = np.sum(r > b)
            blue_wins = np.sum(b > r)
            ties = self.num_simulations - red_wins - blue_wins

            match_results.append(MatchResult(
                match=match,
                red_win_pct=red_wins / self.num_simulations * 100,
                blue_win_pct=blue_wins / self.num_simulations * 100,
                tie_pct=ties / self.num_simulations * 100,
                avg_red_score=float(np.mean(r)),
                avg_blue_score=float(np.mean(b)),
                std_red_score=float(np.std(r)),
                std_blue_score=float(np.std(b)),
            ))

        # ------------------------------------------------------------------
        # Aggregate ranking distributions
        # ------------------------------------------------------------------
        # FTC uses Ranking Points: 2 for win, 1 for tie, 0 for loss
        # Tie-breaker: total points scored across all matches
        ranking_dists = {t: TeamRankingDist(number=t) for t in self.all_teams}

        # Per-simulation accumulators
        team_rp = np.zeros((self.num_simulations, len(self.all_teams)))
        team_tbp = np.zeros((self.num_simulations, len(self.all_teams)))
        team_idx = {t: i for i, t in enumerate(sorted(self.all_teams))}

        for j, match in enumerate(self.schedule):
            r = red_scores[:, j]
            b = blue_scores[:, j]
            red_win_mask = r > b
            blue_win_mask = b > r
            tie_mask = ~red_win_mask & ~blue_win_mask

            for t in match.red_teams:
                idx = team_idx[t]
                team_rp[:, idx] += red_win_mask * 2 + tie_mask * 1
                team_tbp[:, idx] += r

            for t in match.blue_teams:
                idx = team_idx[t]
                team_rp[:, idx] += blue_win_mask * 2 + tie_mask * 1
                team_tbp[:, idx] += b

        # Compute ranks per simulation (higher RP first, then higher TBP)
        # We rank by (-RP, -TBP) so argsort gives ascending rank
        sorted_teams = sorted(self.all_teams)
        for sim_i in range(self.num_simulations):
            rp_vals = team_rp[sim_i]
            tbp_vals = team_tbp[sim_i]
            # Create sort keys: primary = -RP, secondary = -TBP
            order = np.lexsort((-tbp_vals, -rp_vals))
            for rank_0, team_i in enumerate(order):
                t = sorted_teams[team_i]
                rank = rank_0 + 1
                dist = ranking_dists[t]
                dist.rank_counts[rank] = dist.rank_counts.get(rank, 0) + 1

        # Average RP, TBP, and rank
        n = self.num_simulations
        for t in sorted_teams:
            idx = team_idx[t]
            dist = ranking_dists[t]
            dist.avg_rp = float(np.mean(team_rp[:, idx]))
            dist.avg_tbp = float(np.mean(team_tbp[:, idx]))
            # Average rank from distribution
            total_rank = sum(r * c for r, c in dist.rank_counts.items())
            dist.avg_rank = total_rank / n

        # Win/loss/tie counts per team across matches
        for j, match in enumerate(self.schedule):
            r = red_scores[:, j]
            b = blue_scores[:, j]
            rw = np.sum(r > b)
            bw = np.sum(b > r)
            ties = n - rw - bw
            for t in match.red_teams:
                ranking_dists[t].win_count += rw / n
                ranking_dists[t].loss_count += bw / n
                ranking_dists[t].tie_count += ties / n
            for t in match.blue_teams:
                ranking_dists[t].win_count += bw / n
                ranking_dists[t].loss_count += rw / n
                ranking_dists[t].tie_count += ties / n

        return match_results, ranking_dists
