# app.py — Amazon Forecast API (with trends, seasonality heatmaps, smoothing, compare)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Tuple, Dict
import io
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from pmdarima import auto_arima
    PMDARIMA = True
except Exception:
    PMDARIMA = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- helpers ----------
def read_series_from_upload(file: UploadFile) -> Tuple[pd.Series, str, str]:
    """Detect datetime column and numeric target automatically."""
    data = file.file.read()
    df = pd.read_csv(io.BytesIO(data))

    dt_col = None
    # Try to find a date-like column
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(10)
        if sample.str.match(r"\d{4}-\d{2}-\d{2}").all():
            dt_col = c
            break
    if dt_col is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c].dropna().head(5), errors="raise")
                dt_col = c
                break
            except Exception:
                continue

    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)
        df = df.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
    else:
        # fallback to range index
        df.index = pd.RangeIndex(len(df))

    # find numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols and len(df.columns) > 1:
        df[df.columns[1]] = pd.to_numeric(df[df.columns[1]], errors="coerce")
        num_cols = [df.columns[1]]

    target_col = num_cols[0]
    y = pd.to_numeric(df[target_col], errors="coerce").dropna()

    # ensure frequency
    try:
        freq = pd.infer_freq(y.index)
        if freq:
            y = y.asfreq(freq).ffill()
    except Exception:
        pass

    return y, dt_col, target_col


def split_series(y: pd.Series, test_frac: float = 0.2):
    n = len(y)
    h = max(1, int(np.floor(n * test_frac)))
    return y.iloc[:-h], y.iloc[-h:]


def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    )
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def to_series_dict(s: pd.Series):
    return {str(k): float(v) for k, v in s.items()}


# ---------- endpoints ----------

@app.post("/diagnostics")
async def diagnostics(file: UploadFile = File(...), max_lags: int = Form(40)):
    y, dt_col, target_col = read_series_from_upload(file)
    lags = min(max_lags, max(5, len(y) // 4))
    acf_vals = sm_acf(y.dropna(), nlags=lags, fft=True)
    pacf_vals = sm_pacf(y.dropna(), nlags=lags, method="ywm")
    return {
        "info": {"date_col": dt_col, "target_col": target_col, "n": len(y)},
        "acf": [{"lag": i, "value": float(v)} for i, v in enumerate(acf_vals)],
        "pacf": [{"lag": i, "value": float(v)} for i, v in enumerate(pacf_vals)],
    }


@app.post("/auto")
async def auto_select(file: UploadFile = File(...), seasonal_m: int = Form(12)):
    y, _, _ = read_series_from_upload(file)
    y = y.dropna()
    best_arima = {"order": [1, 1, 1], "aic": None}
    best_sarima = {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, seasonal_m], "aic": None}
    if PMDARIMA:
        try:
            ar = auto_arima(
                y,
                seasonal=False,
                stepwise=True,
                error_action="ignore",
                suppress_warnings=True,
            )
            best_arima["order"] = list(ar.order)
            best_arima["aic"] = float(ar.aic())
        except Exception:
            pass
        try:
            sar = auto_arima(
                y,
                seasonal=True,
                m=seasonal_m,
                stepwise=True,
                error_action="ignore",
                suppress_warnings=True,
            )
            best_sarima["order"] = list(sar.order)
            best_sarima["seasonal_order"] = list(sar.seasonal_order)
            best_sarima["aic"] = float(sar.aic())
        except Exception:
            pass
    return {"best_arima": best_arima, "best_sarima": best_sarima, "pmdarima": PMDARIMA}


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    model: str = Form("ARIMA"),
    steps: int = Form(30),
    p: int = Form(1),
    d: int = Form(1),
    q: int = Form(1),
    P: int = Form(1),
    D: int = Form(1),
    Q: int = Form(1),
    m: int = Form(12),
    test_frac: float = Form(0.2),
    transform: str = Form("none"),  # "none" or "log"
    compare: bool = Form(False),  # run both ARIMA and SARIMA
    smoothing: int = Form(7),  # moving average smoothing window for trend
):
    y, _, _ = read_series_from_upload(file)
    y = y.dropna()

    # optional transform
    forward = lambda s: s
    inverse = lambda s: s
    if transform.lower() == "log":
        forward = lambda s: np.log1p(s.clip(lower=0))
        inverse = lambda s: np.expm1(s)
    yt = pd.Series(forward(y), index=y.index)

    train, test = split_series(yt, test_frac)

    def fit_predict(kind, order, seasonal_order=None):
        if kind == "SARIMA":
            mdl = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        else:
            mdl = ARIMA(train, order=order).fit()

        pred = mdl.get_forecast(steps=len(test))
        pred_mean = pd.Series(pred.predicted_mean, index=test.index)
        ci = pred.conf_int(alpha=0.2)
        lower = pd.Series(ci.iloc[:, 0].values, index=test.index)
        upper = pd.Series(ci.iloc[:, 1].values, index=test.index)

        # Full refit for future forecast
        if kind == "SARIMA":
            full = SARIMAX(
                yt,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        else:
            full = ARIMA(yt, order=order).fit()
        fut = full.get_forecast(steps=steps)
        fut_mean = fut.predicted_mean
        fut_ci = fut.conf_int(alpha=0.2)
        fut_lower = pd.Series(fut_ci.iloc[:, 0].values, index=fut_mean.index)
        fut_upper = pd.Series(fut_ci.iloc[:, 1].values, index=fut_mean.index)

        met = metrics(inverse(test), inverse(pred_mean))

        return {
            "order": list(order),
            "seasonal_order": list(seasonal_order) if seasonal_order else None,
            "metrics": met,
            "holdout": {
                "mean": to_series_dict(inverse(pred_mean)),
                "lower": to_series_dict(inverse(lower)),
                "upper": to_series_dict(inverse(upper)),
            },
            "future": {
                "mean": to_series_dict(inverse(pd.Series(fut_mean))),
                "lower": to_series_dict(inverse(fut_lower)),
                "upper": to_series_dict(inverse(fut_upper)),
            },
        }

    result = {"actual_tail": to_series_dict(y.tail(300))}

    if compare:
        result["ARIMA"] = fit_predict("ARIMA", (p, d, q))
        result["SARIMA"] = fit_predict("SARIMA", (p, d, q), (P, D, Q, m))
    else:
        if model.upper() == "SARIMA":
            result["SARIMA"] = fit_predict("SARIMA", (p, d, q), (P, D, Q, m))
        else:
            result["ARIMA"] = fit_predict("ARIMA", (p, d, q))

    # ---------- Trend extraction ----------
    if smoothing > 1:
        trend = y.rolling(window=smoothing, center=True, min_periods=1).mean()
    else:
        trend = y.copy()
    result["trend_tail"] = to_series_dict(trend.tail(300))

    # ---------- Seasonality ----------
    if isinstance(y.index, pd.DatetimeIndex):
        weekday_avg = y.groupby(y.index.day_name()).mean()
        month_avg = y.groupby(y.index.month_name()).mean()
        # reorder by calendar
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekday_avg = weekday_avg.reindex(weekday_order)
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        month_avg = month_avg.reindex(month_order)
        result["weekday"] = weekday_avg.to_dict()
        result["month"] = month_avg.to_dict()

    return result
