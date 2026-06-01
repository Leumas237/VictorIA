"""
tests/test_weather_fetcher.py – Unit tests for weather integration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.weather_fetcher import WeatherFetcher


def test_impact_penalizes_rain_and_wind():
    impact = WeatherFetcher._impact(
        temp_c=10.0,
        humidity=88.0,
        wind_kmh=30.0,
        weather_main="Rain",
        rain_mm=1.2,
    )
    assert impact["confidence_adjustment"] < 0
    assert impact["defense_boost"] > 0
    assert impact["precision_penalty"] > 0


def test_get_weather_without_api_key():
    fetcher = WeatherFetcher(api_key="")
    weather = fetcher.get_weather("Paris")
    assert weather["available"] is False
    assert weather["reason"] == "OPENWEATHER_API_KEY manquante"
