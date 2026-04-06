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
