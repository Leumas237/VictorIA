"""
weather_fetcher.py – OpenWeatherMap integration + impact scoring for VictorIA.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests


class WeatherFetcher:
    """Fetches weather conditions and computes their impact on prediction confidence."""

    GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
    WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
    DEFAULT_TIMEOUT = 5  # keep UI responsive if provider is slow.
    BASE_WEATHER_SCORE = 70  # neutral weather baseline for confidence scoring (0-100).
    ADJUSTMENT_MULTIPLIER = 6  # scales confidence adjustment points into score points.

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = os.getenv("OPENWEATHER_API_KEY", "") if api_key is None else api_key

    @staticmethod
    def _emoji(main: str) -> str:
        mapping = {
            "Clear": "☀️",
            "Clouds": "☁️",
            "Rain": "☔",
            "Drizzle": "🌦️",
            "Thunderstorm": "⛈️",
            "Snow": "❄️",
            "Mist": "🌫️",
            "Fog": "🌫️",
            "Haze": "🌫️",
            "Wind": "🌬️",
        }
        return mapping.get(main, "🌤️")

    @staticmethod
    def _impact(
        temp_c: float,
        humidity: float,
        wind_kmh: float,
        weather_main: str,
        rain_mm: float,
    ) -> dict[str, Any]:
        confidence_adjustment = 0.0
        defense_boost = 0.0
        precision_penalty = 0.0
        notes: list[str] = []

        rainy = weather_main in {"Rain", "Drizzle", "Thunderstorm"} or rain_mm > 0
        if rainy:
            defense_boost += 0.12
            confidence_adjustment -= 2.0
            notes.append("Pluie: rythme plus fermé, avantage défensif.")

        if wind_kmh >= 25:
            precision_penalty += 0.14
            confidence_adjustment -= 2.0
            notes.append("Vent fort: précision technique en baisse.")
        elif wind_kmh >= 15:
            precision_penalty += 0.08
            confidence_adjustment -= 1.0
            notes.append("Vent modéré: légère baisse de précision.")

        if humidity >= 85:
            confidence_adjustment -= 0.5
            notes.append("Humidité élevée: intensité potentiellement réduite.")

        if 12 <= temp_c <= 24 and not rainy and wind_kmh < 15:
            confidence_adjustment += 1.5
            notes.append("Conditions stables: lecture du match plus fiable.")

        # Clamp weather adjustment to avoid over-penalizing or over-boosting model confidence.
        confidence_adjustment = max(-8.0, min(4.0, confidence_adjustment))
        weather_score = round(
            max(
                0.0,
                min(
                    100.0,
                    WeatherFetcher.BASE_WEATHER_SCORE
                    + confidence_adjustment * WeatherFetcher.ADJUSTMENT_MULTIPLIER,
                ),
            ),
            1,
        )

        return {
            "confidence_adjustment": round(confidence_adjustment, 1),
            "weather_confidence_score": weather_score,
            "defense_boost": round(defense_boost, 2),
            "precision_penalty": round(precision_penalty, 2),
            "impact_notes": notes or ["Impact météo limité sur le modèle."],
        }

    def get_weather(self, location_query: str) -> dict[str, Any]:
        """Returns weather payload + impact scoring. Gracefully degrades if unavailable."""
        if not self._api_key:
            return {
                "available": False,
                "location": location_query,
                "reason": "OPENWEATHER_API_KEY manquante",
            }

        try:
            geo_resp = requests.get(
                self.GEO_URL,
                params={"q": location_query, "limit": 1, "appid": self._api_key},
                timeout=self.DEFAULT_TIMEOUT,
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json() or []
            if not geo_data:
                return {
                    "available": False,
                    "location": location_query,
                    "reason": "Localisation introuvable",
                }

            place = geo_data[0]
            weather_resp = requests.get(
                self.WEATHER_URL,
                params={
                    "lat": place["lat"],
                    "lon": place["lon"],
                    "appid": self._api_key,
                    "units": "metric",
                    "lang": "fr",
                },
                timeout=self.DEFAULT_TIMEOUT,
            )
            weather_resp.raise_for_status()
            payload = weather_resp.json()
        except requests.RequestException as exc:
            return {
                "available": False,
                "location": location_query,
                "reason": str(exc),
            }

        main = payload.get("main", {})
        wind = payload.get("wind", {})
        weather = (payload.get("weather") or [{}])[0]

        temp_c = float(main.get("temp", 0.0))
        humidity = float(main.get("humidity", 0.0))
        wind_kmh = float(wind.get("speed", 0.0)) * 3.6
        weather_main = weather.get("main", "Unknown")
        weather_desc = weather.get("description", weather_main)
        rain_mm = float((payload.get("rain") or {}).get("1h", 0.0))

        impact = self._impact(
            temp_c=temp_c,
            humidity=humidity,
            wind_kmh=wind_kmh,
            weather_main=weather_main,
            rain_mm=rain_mm,
        )

        return {
            "available": True,
            "location": ", ".join(filter(None, [place.get("name"), place.get("country")])),
            "emoji": self._emoji(weather_main),
            "conditions": weather_desc.capitalize(),
            "temperature_c": round(temp_c, 1),
            "humidity_pct": round(humidity, 1),
            "wind_kmh": round(wind_kmh, 1),
            "rain_mm": round(rain_mm, 1),
            **impact,
        }
