"""
stats_builder.py – Compute team / H2H stats from match history entries.
Shared by live fetcher and historical dataset builder.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import config

FORM_WEIGHTS = [0.05, 0.08, 0.10, 0.15, 0.62]


def match_entry_for_team(
    home_id: int,
    away_id: int,
    home_goals: int,
    away_goals: int,
    team_id: int,
    utc_date: Optional[str] = None,
    opponent_strength: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """Single match result from one team's perspective."""
    is_home = team_id == home_id
    scored = home_goals if is_home else away_goals
    conceded = away_goals if is_home else home_goals
    if scored > conceded:
        points = 3
    elif scored == conceded:
        points = 1
    else:
        points = 0
    kickoff_hour = _parse_kickoff_hour(utc_date)
    return {
        "scored": scored,
        "conceded": conceded,
        "points": points,
        "is_home": 1 if is_home else 0,
        "is_night": 1 if (kickoff_hour is not None and kickoff_hour >= config.NIGHT_MATCH_START_HOUR) else 0,
        "opponent_strength": float(opponent_strength),
    }


def compute_team_stats(
    entries: List[Dict[str, Union[float, int]]], window: int = 10
) -> Dict[str, Union[float, int, List[int]]]:
    """Aggregate recent match entries into team stats dict."""
    recent = entries[-window:]
    wins = draws = losses = 0
    gf = ga = 0
    form_scores = []
    difficulty_scores = []
    underdog_points = 0
    day_points = day_n = 0
    night_points = night_n = 0
    home_gd = home_n = away_gd = away_n = 0

    for entry in recent:
        gf += entry["scored"]
        ga += entry["conceded"]
        pts = entry["points"]
        form_scores.append(pts)
        opp_strength = float(entry.get("opponent_strength", 0.5))
        difficulty_scores.append(opp_strength)
        if opp_strength >= 0.55 and pts > 0:
            underdog_points += pts
        if entry.get("is_night", 0):
            night_points += pts
            night_n += 1
        else:
            day_points += pts
            day_n += 1
        goal_diff = entry["scored"] - entry["conceded"]
        if entry.get("is_home", 0):
            home_gd += goal_diff
            home_n += 1
        else:
            away_gd += goal_diff
            away_n += 1
        if pts == 3:
            wins += 1
        elif pts == 1:
            draws += 1
        else:
            losses += 1

    n = max(len(recent), 1)
    last5 = form_scores[-5:]
    recent_form = (
        sum(w * s for w, s in zip(FORM_WEIGHTS, last5))
        if last5
        else 1.5
    )
    window_average_form = (sum(form_scores) / n) if form_scores else 1.5
    form_score = (
        config.RECENT_FORM_WEIGHT * recent_form +
        config.HISTORICAL_FORM_WEIGHT * window_average_form
    )
    fixture_difficulty = float(sum(difficulty_scores) / max(len(difficulty_scores), 1))
    underdog_factor = float(
        underdog_points / max(config.MAX_POINTS_PER_MATCH * len(recent), 1)
    )
    clean_sheet_streak = _streak_length(recent, lambda e: e["conceded"] == 0)
    scoring_streak = _streak_length(recent, lambda e: e["scored"] > 0)
    scoreless_streak = _streak_length(recent, lambda e: e["scored"] == 0)

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_scored": gf,
        "goals_conceded": ga,
        "avg_goals_scored": gf / n,
        "avg_goals_conceded": ga / n,
        "win_rate": wins / n,
        "form_score": form_score,
        "recent_form_score": recent_form,
        "historical_form_score": window_average_form,
        "form_last5": last5,
        "matches_played": n,
        "fixture_difficulty_recent": fixture_difficulty,
        "difficulty_adjusted_form": form_score / max(0.25, fixture_difficulty),
        "clean_sheet_streak": clean_sheet_streak,
        "scoring_streak": scoring_streak,
        "scoreless_streak": scoreless_streak,
        "day_points_per_game": day_points / max(day_n, 1),
        "night_points_per_game": night_points / max(night_n, 1),
        "day_night_points_diff": (day_points / max(day_n, 1)) - (night_points / max(night_n, 1)),
        "home_goal_diff_avg": home_gd / max(home_n, 1),
        "away_goal_diff_avg": away_gd / max(away_n, 1),
        "home_away_goal_diff_gap": (home_gd / max(home_n, 1)) - (away_gd / max(away_n, 1)),
        "underdog_factor": underdog_factor,
    }


def compute_h2h(h2h_results: List[Tuple[int, int]]) -> Dict:
    """
    h2h_results: list of (home_goals, away_goals) for the upcoming fixture's
    home/away orientation (before the current match).
    """
    home_wins = draws = away_wins = 0
    for hg, ag in h2h_results:
        if hg > ag:
            home_wins += 1
        elif hg < ag:
            away_wins += 1
        else:
            draws += 1
    total = len(h2h_results)
    return {
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "total": total,
    }


def match_outcome_label(home_goals: int, away_goals: int) -> int:
    """0=HomeWin, 1=Draw, 2=AwayWin."""
    if home_goals > away_goals:
        return 0
    if home_goals < away_goals:
        return 2
    return 1


def _parse_kickoff_hour(utc_date: Optional[str]) -> Optional[int]:
    if not utc_date:
        return None
    try:
        return datetime.fromisoformat(utc_date.replace("Z", "+00:00")).hour
    except (ValueError, TypeError):
        return None


def _streak_length(entries: List[Dict[str, Union[float, int]]], predicate) -> int:
    streak = 0
    for entry in reversed(entries):
        if predicate(entry):
            streak += 1
        else:
            break
    return streak
