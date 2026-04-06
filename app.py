from __future__ import annotations

import os
import pickle
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import holidays
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", BASE_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "load_shape_model.pkl")
TREND_PATH = os.path.join(MODEL_DIR, "trend_df.pkl")
CLIMATOLOGY_PATH = os.path.join(MODEL_DIR, "climatology.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.pkl")
WEATHER_URL_TEMPLATE = os.getenv(
    "WEATHER_URL_TEMPLATE",
    "http://openaccess.pf.api.met.ie/metno-wdb2ts/locationforecast?lat={lat};long={lon}",
)

app = FastAPI(title="Power Forecast API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Any = None
trend_df: pd.DataFrame | None = None
climatology: dict[str, Any] | None = None
meta: dict[str, Any] | None = None
REFERENCE_YEAR: int | None = None
IE_HOLIDAYS = holidays.country_holidays("IE")

NORMAL_CODES = {0, 1, 2, 3, 4, 5, 25, 50, 51, 9, 28, 60}
MODERATE_CODES = {
    6, 40, 41, 42, 44, 52, 53, 72, 85, 10, 58, 61, 46, 12, 63, 80,
    30, 71, 17, 66, 70, 18, 76, 14, 68, 15, 32, 83, 78, 79, 87,
}
SEVERE_CODES = {
    7, 8, 43, 45, 47, 48, 49, 11, 29, 59, 62, 13, 81, 82, 31, 67,
    73, 19, 77, 86, 16, 33, 69, 84, 74, 75, 88, 26, 54, 55, 27, 56,
    64, 20, 95, 21, 91, 23, 24, 34, 35, 93, 89,
}
EXTREME_CODES = {57, 65, 22, 92, 94, 96, 97, 98, 99, 90}

dashboard_cache: dict[str, Any] | None = None
dashboard_cache_time: datetime | None = None
DASHBOARD_CACHE_TTL = timedelta(minutes=5)


class HealthResponse(BaseModel):
    status: str
    generated_at: str


class HourlyPoint(BaseModel):
    time: str
    value: float


class DashboardResponse(BaseModel):
    next_hour: float
    tomorrow_total: float
    next_7_days: float
    peak_time: str
    peak_value: float
    hourly: list[float]
    hourly_points: list[HourlyPoint]
    generated_at: str


class ForecastPoint(BaseModel):
    datetime: str
    forecast: float
    temp: float | None = None
    symbol_name: str | None = None
    weather_classification: str | None = None
    weekday: str | None = None
    holiday: str | None = None
    season: str | None = None


class ForecastResponse(BaseModel):
    hours: int
    lat: float
    lon: float
    generated_at: str
    points: list[ForecastPoint]


class DailySummaryPoint(BaseModel):
    date: str
    forecast: float
    temp: float | None = None
    symbol_name: str | None = None
    weather_classification: str | None = None
    weekday: str | None = None
    holiday: str | None = None
    season: str | None = None


class DailySummaryResponse(BaseModel):
    hours: int
    lat: float
    lon: float
    generated_at: str
    daily: list[DailySummaryPoint]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


@app.on_event("startup")
def startup_event() -> None:
    global model, trend_df, climatology, meta, REFERENCE_YEAR
    model = load_pickle(MODEL_PATH)
    trend_df = load_pickle(TREND_PATH)
    climatology = load_pickle(CLIMATOLOGY_PATH)
    meta = load_pickle(META_PATH)
    REFERENCE_YEAR = int(meta["reference_year"])


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Power Forecast API is running.",
        "health": "/health",
        "dashboard": "/dashboard",
        "forecast": "/forecast?hours=24",
        "daily": "/daily-summary?hours=168",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", generated_at=now_utc())


def map_symbol_number(code: Any) -> str:
    if pd.isna(code):
        return "Unknown"
    try:
        code = int(code)
    except Exception:
        return "Unknown"

    if code in NORMAL_CODES:
        return "Normal"
    if code in MODERATE_CODES:
        return "Moderate"
    if code in SEVERE_CODES:
        return "Severe"
    if code in EXTREME_CODES:
        return "Extreme"
    return "Unknown"


def season_label(month: int) -> str:
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    if month in [9, 10, 11]:
        return "Autumn"
    return "Winter"


def weekday_name(ts: pd.Timestamp) -> str:
    return ts.strftime("%A")


def weekday_type(ts: pd.Timestamp) -> str:
    return "Weekend" if ts.dayofweek >= 5 else "Weekday"


def holiday_flag(ts: pd.Timestamp) -> int:
    return int(ts.date() in IE_HOLIDAYS)


def holiday_name(ts: pd.Timestamp) -> str:
    return IE_HOLIDAYS.get(ts.date(), "No")


def trend_factor(year: int) -> float:
    assert trend_df is not None
    row = trend_df[trend_df["year"] == year]
    if len(row) > 0:
        return float(row.iloc[0]["trend_factor_vs_reference_year"])
    return float(trend_df.iloc[-1]["trend_factor_vs_reference_year"])


def time_features(ts: pd.Timestamp) -> dict[str, float | int]:
    h = ts.hour
    d = ts.dayofweek
    m = ts.month
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * d / 7),
        "dow_cos": np.sin(2 * np.pi * d / 7 + np.pi / 2),  # preserves original dimensionality
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
        "is_monday": int(d == 0),
        "is_tuesday": int(d == 1),
        "is_wednesday": int(d == 2),
        "is_thursday": int(d == 3),
        "is_friday": int(d == 4),
        "is_saturday": int(d == 5),
        "is_sunday": int(d == 6),
    }


def _to_naive_datetime(x: str) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        return ts
    return ts.tz_convert(None)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1].lower()
    return tag.lower()


def _safe_int(x: Any) -> float:
    try:
        return int(float(x))
    except Exception:
        return np.nan


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _extract_symbol_name(attrs: dict[str, Any], symbol_number: float = np.nan) -> str:
    for key in ["name", "id", "var", "symbolid", "code", "description", "desc", "numberEx"]:
        if key in attrs and attrs[key] not in [None, ""]:
            return str(attrs[key])
    if pd.notna(symbol_number):
        return f"Code_{int(symbol_number)}"
    return "Unknown"


def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    try:
        url = WEATHER_URL_TEMPLATE.format(lat=lat, lon=lon)
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.text)

        instant_rows: list[dict[str, Any]] = []
        symbol_rows: list[dict[str, Any]] = []

        for t in root.iter():
            if _strip_ns(t.tag) != "time":
                continue

            from_attr = t.attrib.get("from")
            if not from_attr:
                continue

            from_ts = _to_naive_datetime(from_attr)
            temp = np.nan
            symbol_number = np.nan
            symbol_name = None

            for c in t.iter():
                tag_c = _strip_ns(c.tag)

                if tag_c == "temperature" and "value" in c.attrib:
                    temp = _safe_float(c.attrib.get("value"))

                if "symbol" in tag_c:
                    raw_number = None
                    for num_key in ["number", "numberex", "code"]:
                        if num_key in c.attrib:
                            raw_number = c.attrib.get(num_key)
                            break

                    parsed_number = _safe_int(raw_number)
                    if pd.notna(parsed_number) and pd.isna(symbol_number):
                        symbol_number = parsed_number

                    parsed_name = _extract_symbol_name(c.attrib, parsed_number)
                    if symbol_name is None or symbol_name == "Unknown":
                        symbol_name = parsed_name

            if (symbol_name is None or symbol_name == "Unknown") and pd.notna(symbol_number):
                symbol_name = f"Code_{int(symbol_number)}"

            if pd.notna(temp):
                instant_rows.append({"datetime": from_ts, "temp": temp})

            if pd.notna(symbol_number) or symbol_name is not None:
                symbol_rows.append(
                    {
                        "datetime": from_ts,
                        "symbol_number": symbol_number,
                        "symbol_name": symbol_name,
                    }
                )

        instant_df = pd.DataFrame(instant_rows)
        symbol_df = pd.DataFrame(symbol_rows)

        if len(instant_df) == 0 and len(symbol_df) == 0:
            return pd.DataFrame()

        if len(instant_df) > 0:
            instant_df = (
                instant_df.dropna(subset=["datetime"])
                .drop_duplicates(subset=["datetime"], keep="first")
                .sort_values("datetime")
            )

        if len(symbol_df) > 0:
            symbol_df = (
                symbol_df.dropna(subset=["datetime"])
                .drop_duplicates(subset=["datetime"], keep="first")
                .sort_values("datetime")
            )

        if len(instant_df) > 0 and len(symbol_df) > 0:
            df = pd.merge_asof(
                instant_df.sort_values("datetime"),
                symbol_df.sort_values("datetime"),
                on="datetime",
                direction="nearest",
            )
        elif len(instant_df) > 0:
            df = instant_df.copy()
            df["symbol_number"] = np.nan
            df["symbol_name"] = "Unknown"
        else:
            df = symbol_df.copy()
            df["temp"] = np.nan

        df["temp"] = pd.to_numeric(df["temp"], errors="coerce").interpolate(limit_direction="both")
        df["weather_classification"] = df["symbol_number"].apply(map_symbol_number)

        def _final_symbol_name(row: pd.Series) -> str:
            if pd.notna(row["symbol_name"]) and str(row["symbol_name"]).strip() not in ["", "None", "nan", "Unknown"]:
                return str(row["symbol_name"])
            if pd.notna(row["symbol_number"]):
                return f"Code_{int(row['symbol_number'])}"
            return "Unknown"

        df["symbol_name"] = df.apply(_final_symbol_name, axis=1)
        return df.set_index("datetime").sort_index()

    except Exception:
        return pd.DataFrame()


def climatology_history(hours: int = 168) -> pd.DataFrame:
    assert climatology is not None
    prof = climatology["demand_profile"].copy()
    idx = pd.date_range(end=pd.Timestamp.now().tz_localize(None), periods=hours, freq="H")

    vals: list[float] = []
    for t in idx:
        hflag = holiday_flag(t)
        row = prof[
            (prof["month"] == t.month)
            & (prof["dow"] == t.dayofweek)
            & (prof["hour"] == t.hour)
            & (prof["holiday"] == hflag)
        ]
        if len(row) > 0:
            val = row["demand_kwh"].values[0]
        else:
            row2 = prof[
                (prof["month"] == t.month)
                & (prof["dow"] == t.dayofweek)
                & (prof["hour"] == t.hour)
            ]
            if len(row2) > 0:
                val = row2["demand_kwh"].median()
            else:
                val = prof["demand_kwh"].median()
        vals.append(float(val))

    return pd.DataFrame({"demand_kwh": vals}, index=idx)


def prepare_weather_for_index(weather_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    if weather_df is None or len(weather_df) == 0:
        return pd.DataFrame(index=target_index)

    w = weather_df.copy()
    w = w[~w.index.duplicated(keep="first")].sort_index()
    union_index = w.index.union(target_index)
    w = w.reindex(union_index).sort_index()

    if "temp" in w.columns:
        w["temp"] = pd.to_numeric(w["temp"], errors="coerce").interpolate(method="time").ffill().bfill()

    for col in ["symbol_number", "symbol_name", "weather_classification"]:
        if col in w.columns:
            w[col] = w[col].ffill().bfill()

    return w.reindex(target_index)


def get_forecast_start(hours: int) -> pd.Timestamp:
    now = pd.Timestamp.now().tz_localize(None)
    if hours in [1, 24]:
        return now.floor("H") + pd.Timedelta(hours=1)
    return now.normalize() + pd.Timedelta(days=1)


def get_climatology_anchor(ts: pd.Timestamp) -> float:
    assert climatology is not None
    prof = climatology["demand_profile"].copy()
    hflag = holiday_flag(ts)

    row = prof[
        (prof["month"] == ts.month)
        & (prof["dow"] == ts.dayofweek)
        & (prof["hour"] == ts.hour)
        & (prof["holiday"] == hflag)
    ]
    if len(row) > 0:
        return float(row["demand_kwh"].median())

    row2 = prof[(prof["month"] == ts.month) & (prof["dow"] == ts.dayofweek) & (prof["hour"] == ts.hour)]
    if len(row2) > 0:
        return float(row2["demand_kwh"].median())

    row3 = prof[prof["hour"] == ts.hour]
    if len(row3) > 0:
        return float(row3["demand_kwh"].median())

    return float(prof["demand_kwh"].median())


def build_feature_row(ts: pd.Timestamp, history: pd.DataFrame, weather_row: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    tf = time_features(ts)
    holiday_flag_val = holiday_flag(ts)
    non_holiday_val = 1 - holiday_flag_val

    weekday_val = weekday_name(ts)
    weekday_type_val = weekday_type(ts)
    season_val = season_label(ts.month)

    temp = 10.0
    wc = "Unknown"
    symbol_number = np.nan
    symbol_name = "Unknown"

    if weather_row is not None and len(weather_row) > 0:
        if "temp" in weather_row and pd.notna(weather_row["temp"]):
            temp = float(weather_row["temp"])
        if "weather_classification" in weather_row and pd.notna(weather_row["weather_classification"]):
            wc = str(weather_row["weather_classification"])
        if "symbol_number" in weather_row and pd.notna(weather_row["symbol_number"]):
            symbol_number = weather_row["symbol_number"]
        if "symbol_name" in weather_row and pd.notna(weather_row["symbol_name"]):
            symbol_name = str(weather_row["symbol_name"])

    lag1 = float(history.iloc[-1, 0])
    lag24 = float(history.iloc[-24, 0]) if len(history) >= 24 else lag1
    lag168 = float(history.iloc[-168, 0]) if len(history) >= 168 else lag24
    roll_mean_3 = float(history.iloc[-3:, 0].mean()) if len(history) >= 3 else lag1
    roll_mean_24 = float(history.iloc[-24:, 0].mean()) if len(history) >= 24 else lag1
    roll_mean_168 = float(history.iloc[-168:, 0].mean()) if len(history) >= 168 else lag1
    roll_std_24 = float(history.iloc[-24:, 0].std()) if len(history) >= 24 else 0.0
    roll_std_168 = float(history.iloc[-168:, 0].std()) if len(history) >= 168 else 0.0

    row = {
        "temp": temp,
        "weather_classification": wc,
        "weekday_type": weekday_type_val,
        "season": season_val,
        "dow_name": weekday_val,
        "holiday": holiday_flag_val,
        "hour_sin": tf["hour_sin"],
        "hour_cos": tf["hour_cos"],
        "dow_sin": tf["dow_sin"],
        "dow_cos": tf["dow_cos"],
        "month_sin": tf["month_sin"],
        "month_cos": tf["month_cos"],
        "is_monday": tf["is_monday"],
        "is_tuesday": tf["is_tuesday"],
        "is_wednesday": tf["is_wednesday"],
        "is_thursday": tf["is_thursday"],
        "is_friday": tf["is_friday"],
        "is_saturday": tf["is_saturday"],
        "is_sunday": tf["is_sunday"],
        "lag_1": lag1,
        "lag_24": lag24,
        "lag_168": lag168,
        "roll_mean_3": roll_mean_3,
        "roll_mean_24": roll_mean_24,
        "roll_mean_168": roll_mean_168,
        "roll_std_24": roll_std_24,
        "roll_std_168": roll_std_168,
        "holiday_hour": holiday_flag_val * ts.hour,
        "temp_x_holiday": temp * holiday_flag_val,
        "temp_x_nonholiday": temp * non_holiday_val,
        "temp_x_monday": temp * tf["is_monday"],
        "temp_x_tuesday": temp * tf["is_tuesday"],
        "temp_x_wednesday": temp * tf["is_wednesday"],
        "temp_x_thursday": temp * tf["is_thursday"],
        "temp_x_friday": temp * tf["is_friday"],
        "temp_x_saturday": temp * tf["is_saturday"],
        "temp_x_sunday": temp * tf["is_sunday"],
        "holiday_x_monday": holiday_flag_val * tf["is_monday"],
        "holiday_x_tuesday": holiday_flag_val * tf["is_tuesday"],
        "holiday_x_wednesday": holiday_flag_val * tf["is_wednesday"],
        "holiday_x_thursday": holiday_flag_val * tf["is_thursday"],
        "holiday_x_friday": holiday_flag_val * tf["is_friday"],
        "holiday_x_saturday": holiday_flag_val * tf["is_saturday"],
        "holiday_x_sunday": holiday_flag_val * tf["is_sunday"],
        "hour_x_monday": ts.hour * tf["is_monday"],
        "hour_x_tuesday": ts.hour * tf["is_tuesday"],
        "hour_x_wednesday": ts.hour * tf["is_wednesday"],
        "hour_x_thursday": ts.hour * tf["is_thursday"],
        "hour_x_friday": ts.hour * tf["is_friday"],
        "hour_x_saturday": ts.hour * tf["is_saturday"],
        "hour_x_sunday": ts.hour * tf["is_sunday"],
    }

    extra_info = {
        "temp": temp,
        "symbol_number": symbol_number,
        "symbol_name": symbol_name,
        "weather_classification": wc,
        "weekday": weekday_val,
        "weekday_type": weekday_type_val,
        "holiday": holiday_name(ts),
        "season": season_val,
    }
    return pd.DataFrame([row]), extra_info


def forecast(hours: int, lat: float = 53.34, lon: float = -6.26) -> pd.DataFrame:
    if hours <= 0:
        raise ValueError("hours must be > 0")
    if model is None:
        raise RuntimeError("Model is not loaded")

    history = climatology_history()
    weather_raw = fetch_weather(lat, lon)

    start = get_forecast_start(hours)
    idx = pd.date_range(start=start, periods=hours, freq="H")
    weather = prepare_weather_for_index(weather_raw, idx)

    rows: list[dict[str, Any]] = []
    for ts in idx:
        weather_row = weather.loc[ts] if len(weather) > 0 and ts in weather.index else pd.Series(dtype="object")
        x, extra = build_feature_row(ts, history, weather_row)

        raw_pred = float(model.predict(x)[0]) * trend_factor(ts.year)
        lag24_anchor = float(x["lag_24"].iloc[0])
        lag168_anchor = float(x["lag_168"].iloc[0])
        clim_anchor = get_climatology_anchor(ts)

        anchor_lag = 0.5 * lag24_anchor + 0.5 * lag168_anchor
        anchor = 0.55 * anchor_lag + 0.45 * clim_anchor

        y = 0.55 * raw_pred + 0.45 * anchor
        y = np.clip(y, anchor * 0.85, anchor * 1.15)
        y = max(float(y), 0.0)

        rows.append(
            {
                "datetime": ts,
                "forecast": round(y, 2),
                "temp": round(float(extra["temp"]), 2),
                "symbol_number": extra["symbol_number"],
                "symbol_name": extra["symbol_name"],
                "weather_classification": extra["weather_classification"],
                "weekday": extra["weekday"],
                "weekday_type": extra["weekday_type"],
                "holiday": extra["holiday"],
                "season": extra["season"],
            }
        )
        history.loc[ts] = y

    return pd.DataFrame(rows).set_index("datetime")


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.copy()
    daily["date"] = daily.index.date

    def mode_or_first(s: pd.Series) -> Any:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        m = s.mode()
        if len(m) > 0:
            return m.iloc[0]
        return s.iloc[0]

    out = daily.groupby("date").agg(
        {
            "forecast": "sum",
            "temp": "mean",
            "symbol_name": lambda s: mode_or_first(s),
            "weather_classification": lambda s: mode_or_first(s),
            "weekday": lambda s: mode_or_first(s),
            "weekday_type": lambda s: mode_or_first(s),
            "holiday": lambda s: mode_or_first(s),
            "season": lambda s: mode_or_first(s),
        }
    )
    out["temp"] = out["temp"].round(2)
    return out


@app.get("/forecast", response_model=ForecastResponse)
def forecast_endpoint(
    hours: int = Query(24, ge=1, le=24 * 31),
    lat: float = Query(53.34),
    lon: float = Query(-6.26),
) -> ForecastResponse:
    try:
        df = forecast(hours=hours, lat=lat, lon=lon)
        points = [
            ForecastPoint(
                datetime=ts.isoformat(),
                forecast=float(row["forecast"]),
                temp=float(row["temp"]) if pd.notna(row["temp"]) else None,
                symbol_name=str(row["symbol_name"]) if pd.notna(row["symbol_name"]) else None,
                weather_classification=str(row["weather_classification"]) if pd.notna(row["weather_classification"]) else None,
                weekday=str(row["weekday"]) if pd.notna(row["weekday"]) else None,
                holiday=str(row["holiday"]) if pd.notna(row["holiday"]) else None,
                season=str(row["season"]) if pd.notna(row["season"]) else None,
            )
            for ts, row in df.iterrows()
        ]
        return ForecastResponse(hours=hours, lat=lat, lon=lon, generated_at=now_utc(), points=points)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"forecast failed: {exc}") from exc


@app.get("/daily-summary", response_model=DailySummaryResponse)
def daily_summary_endpoint(
    hours: int = Query(24 * 7, ge=24, le=24 * 31),
    lat: float = Query(53.34),
    lon: float = Query(-6.26),
) -> DailySummaryResponse:
    try:
        df = forecast(hours=hours, lat=lat, lon=lon)
        daily_df = summarize_daily(df)
        daily = [
            DailySummaryPoint(
                date=str(idx),
                forecast=round(float(row["forecast"]), 2),
                temp=float(row["temp"]) if pd.notna(row["temp"]) else None,
                symbol_name=str(row["symbol_name"]) if pd.notna(row["symbol_name"]) else None,
                weather_classification=str(row["weather_classification"]) if pd.notna(row["weather_classification"]) else None,
                weekday=str(row["weekday"]) if pd.notna(row["weekday"]) else None,
                holiday=str(row["holiday"]) if pd.notna(row["holiday"]) else None,
                season=str(row["season"]) if pd.notna(row["season"]) else None,
            )
            for idx, row in daily_df.iterrows()
        ]
        return DailySummaryResponse(hours=hours, lat=lat, lon=lon, generated_at=now_utc(), daily=daily)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"daily summary failed: {exc}") from exc


@app.get("/dashboard", response_model=DashboardResponse)
def dashboard(lat: float = Query(53.34), lon: float = Query(-6.26)) -> DashboardResponse:
    global dashboard_cache, dashboard_cache_time

    try:
        now = datetime.now(timezone.utc)

        if (
            dashboard_cache is not None
            and dashboard_cache_time is not None
            and now - dashboard_cache_time < DASHBOARD_CACHE_TTL
        ):
            return DashboardResponse(**dashboard_cache)

        # 只算一次 168 小时，提高速度且保证所有指标来自同一批真实预测结果
        all_df = forecast(hours=24 * 7, lat=lat, lon=lon)
        next_hour_df = all_df.iloc[:1]
        day_df = all_df.iloc[:24]
        week_df = all_df

        hourly = [round(float(v), 1) for v in day_df["forecast"].tolist()]
        hourly_points = [
            {
                "time": ts.isoformat(),
                "value": round(float(row["forecast"]), 1),
            }
            for ts, row in day_df.iterrows()
        ]

        peak_value = max(hourly)
        peak_idx = hourly.index(peak_value)
        peak_ts = day_df.index[peak_idx]
        peak_time = peak_ts.strftime("%H:%M")

        result = {
            "next_hour": round(float(next_hour_df["forecast"].iloc[0]), 1),
            "tomorrow_total": round(float(day_df["forecast"].sum()), 1),
            "next_7_days": round(float(week_df["forecast"].sum()), 1),
            "peak_time": peak_time,
            "peak_value": round(float(peak_value), 1),
            "hourly": hourly,
            "hourly_points": hourly_points,
            "generated_at": now_utc(),
        }

        dashboard_cache = result
        dashboard_cache_time = now

        return DashboardResponse(**result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dashboard failed: {exc}") from exc
