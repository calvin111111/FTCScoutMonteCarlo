"""
Formatting helpers for simulation results.
"""

from simulation import MatchResult, TeamRankingDist

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def _pct_bar(pct: float, width: int = 20) -> str:
    """Simple ASCII bar for a percentage value."""
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def format_match_predictions(results: list[MatchResult]) -> str:
    """Format match predictions as a table."""
    headers = [
        "Match", "Red Teams", "Blue Teams",
        "Red Win%", "Blue Win%", "Tie%",
        "Avg Red", "Avg Blue",
    ]
    rows = []
    for r in results:
        red_teams = ", ".join(str(t) for t in r.match.red_teams)
        blue_teams = ", ".join(str(t) for t in r.match.blue_teams)
        rows.append([
            r.match.match_name,
            red_teams,
            blue_teams,
            f"{r.red_win_pct:5.1f}%",
            f"{r.blue_win_pct:5.1f}%",
            f"{r.tie_pct:4.1f}%",
            f"{r.avg_red_score:.0f} ±{r.std_red_score:.0f}",
            f"{r.avg_blue_score:.0f} ±{r.std_blue_score:.0f}",
        ])

    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="simple")
    else:
        return _fallback_table(headers, rows)


def format_rankings(dists: dict[int, TeamRankingDist],
                    top_n: int = 0) -> str:
    """Format ranking predictions as a table.

    Args:
        dists: team number -> TeamRankingDist
        top_n: If >0, only show the top N teams by average rank.
    """
    sorted_teams = sorted(dists.values(), key=lambda d: d.avg_rank)
    if top_n > 0:
        sorted_teams = sorted_teams[:top_n]

    headers = [
        "Pred Rank", "Team", "Avg RP", "Avg TBP",
        "W", "L", "T", "Top 4%", "Rank Dist",
    ]
    rows = []
    total_sims = sum(dists[list(dists.keys())[0]].rank_counts.values()) \
        if dists else 1

    for i, d in enumerate(sorted_teams, 1):
        # Probability of finishing in top 4
        top4 = sum(d.rank_counts.get(r, 0) for r in range(1, 5))
        top4_pct = top4 / total_sims * 100

        # Compact rank distribution (top 3 most likely ranks)
        top_ranks = sorted(d.rank_counts.items(), key=lambda x: -x[1])[:3]
        rank_dist = ", ".join(
            f"#{r}:{c / total_sims * 100:.0f}%"
            for r, c in top_ranks
        )

        rows.append([
            f"#{i}",
            str(d.number),
            f"{d.avg_rp:.2f}",
            f"{d.avg_tbp:.0f}",
            f"{d.win_count:.1f}",
            f"{d.loss_count:.1f}",
            f"{d.tie_count:.1f}",
            f"{top4_pct:.1f}%",
            rank_dist,
        ])

    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="simple")
    else:
        return _fallback_table(headers, rows)


def _fallback_table(headers: list[str], rows: list[list[str]]) -> str:
    """Simple column-aligned table without tabulate."""
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]
    lines = []
    for row in all_rows:
        line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
        lines.append(line)
        if row is headers:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
