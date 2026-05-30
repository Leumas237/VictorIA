"""
real_dataset.py – Build training dataset from real historical matches.
Features are computed point-in-time (no data leakage).
"""
import json
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from data.historical_fetcher import HistoricalFetcher
from data.preprocessor import Preprocessor
from data.stats_builder import (
    compute_h2h,
    compute_team_stats,
    match_entry_for_team,
    match_outcome_label,
)


def build_real_dataset(
    competitions: Optional[List[str]] = None,
    seasons: Optional[List[int]] = None,
    min_team_matches: int = None,
    stats_window: int = None,
    force_refresh: bool = False,
    min_raw_matches: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build (X, y) from football-data.org historical matches.
    Caches result to config.REAL_DATASET_PATH.
    """
    min_team_matches = min_team_matches or config.MIN_TEAM_MATCHES
    stats_window = stats_window or config.STATS_WINDOW

    if config.REAL_DATASET_PATH.exists() and not force_refresh:
        print(f"[RealDataset] Chargement cache → {config.REAL_DATASET_PATH}")
        with open(config.REAL_DATASET_PATH, "rb") as f:
            cached = pickle.load(f)
        return cached["X"], cached["y"], cached["meta"]

    fetcher = HistoricalFetcher()
    matches, fetch_meta = fetcher.fetch_all(competitions, seasons)

    if len(matches) < min_raw_matches:
        raise ValueError(
            f"Pas assez de matchs réels ({len(matches)}). "
            "Vérifiez votre clé API et les saisons/compétitions disponibles."
        )

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    sample_seasons: List[int] = []
    sample_competitions: List[str] = []

    team_history: Dict[int, List[Dict[str, int]]] = defaultdict(list)
    h2h_history: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

    preprocessor = Preprocessor()
    skipped_insufficient = 0
    skipped_anomalies = 0
    skipped_invalid_date = 0

    for match in matches:
        if not _is_valid_match(match):
            skipped_anomalies += 1
            continue
        if not _is_valid_utc_date(match.get("utcDate", "")):
            skipped_invalid_date += 1
            continue

        home_id = match["homeTeam"]["id"]
        away_id = match["awayTeam"]["id"]
        home_goals = match["homeGoals"]
        away_goals = match["awayGoals"]

        home_ready = len(team_history[home_id]) >= min_team_matches
        away_ready = len(team_history[away_id]) >= min_team_matches

        home_opp_strength = 0.5
        away_opp_strength = 0.5
        if home_ready and away_ready:
            dynamic_home_window = _adaptive_window(len(team_history[home_id]), stats_window)
            dynamic_away_window = _adaptive_window(len(team_history[away_id]), stats_window)
            home_stats = compute_team_stats(
                team_history[home_id], window=dynamic_home_window
            )
            away_stats = compute_team_stats(
                team_history[away_id], window=dynamic_away_window
            )
            home_opp_strength = _team_strength(away_stats)
            away_opp_strength = _team_strength(home_stats)
            h2h = compute_h2h(h2h_history[(home_id, away_id)])

            match_data = {
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "competition": match["competition"],
                "home_stats": home_stats,
                "away_stats": away_stats,
                "h2h": h2h,
            }

            features = preprocessor.extract_features(match_data)
            X_rows.append(features.values[0])
            y_rows.append(match_outcome_label(home_goals, away_goals))
            sample_seasons.append(match["season"])
            sample_competitions.append(match["competition"])
        else:
            skipped_insufficient += 1

        # Update histories AFTER feature extraction (no leakage)
        team_history[home_id].append(
            match_entry_for_team(
                home_id, away_id, home_goals, away_goals, home_id,
                utc_date=match.get("utcDate"),
                opponent_strength=home_opp_strength,
            )
        )
        team_history[away_id].append(
            match_entry_for_team(
                home_id, away_id, home_goals, away_goals, away_id,
                utc_date=match.get("utcDate"),
                opponent_strength=away_opp_strength,
            )
        )
        h2h_history[(home_id, away_id)].append((home_goals, away_goals))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=int)

    meta = {
        **fetch_meta,
        "n_samples": int(len(y)),
        "n_skipped_insufficient_history": skipped_insufficient,
        "n_skipped_anomalies": skipped_anomalies,
        "n_skipped_invalid_date": skipped_invalid_date,
        "min_team_matches": min_team_matches,
        "stats_window": stats_window,
        "seasons": sample_seasons,
        "competitions_per_sample": sample_competitions,
        "label_distribution": {
            "HomeWin": int((y == 0).sum()),
            "Draw": int((y == 1).sum()),
            "AwayWin": int((y == 2).sum()),
        },
    }

    _save_dataset_cache(X, y, meta)
    return X, y, meta


def _adaptive_window(history_len: int, max_window: int) -> int:
    if history_len <= config.MIN_ADAPTIVE_WINDOW:
        return history_len
    return min(
        max_window,
        max(config.MIN_ADAPTIVE_WINDOW, int(history_len * config.ADAPTIVE_HISTORY_RATIO)),
    )


def _is_valid_match(match: Dict[str, Any]) -> bool:
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    home_id = home.get("id")
    away_id = away.get("id")
    if not home_id or not away_id or home_id == away_id:
        return False
    hg = match.get("homeGoals")
    ag = match.get("awayGoals")
    if hg is None or ag is None:
        return False
    if hg < 0 or ag < 0:
        return False
    if hg > config.MAX_VALID_GOALS_PER_TEAM or ag > config.MAX_VALID_GOALS_PER_TEAM:
        return False
    return True


def _is_valid_utc_date(utc_date: str) -> bool:
    try:
        datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


def _team_strength(stats: Dict[str, Any]) -> float:
    return float(
        config.TEAM_STRENGTH_WEIGHT_WIN_RATE * stats.get("win_rate", 0.5) +
        config.TEAM_STRENGTH_WEIGHT_FORM * (
            stats.get("form_score", 1.5) / config.TEAM_STRENGTH_FORM_DIVISOR
        ) +
        config.TEAM_STRENGTH_WEIGHT_GOAL_EDGE * (
            stats.get("avg_goals_scored", 1.0) - stats.get("avg_goals_conceded", 1.0)
        )
    )


def _save_dataset_cache(X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
    with open(config.REAL_DATASET_PATH, "wb") as f:
        pickle.dump({"X": X, "y": y, "meta": meta}, f)
    with open(config.REAL_DATASET_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in meta.items() if k not in ("seasons", "competitions_per_sample")},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[RealDataset] {len(y)} échantillons sauvegardés → {config.REAL_DATASET_PATH}")
