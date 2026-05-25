# The Runway

Predicts whether a US domestic flight will be delayed **before it departs** — using only information available at booking time.

Trained on 5.7 million flights. No data leakage. No post-departure features.

---

## The Problem

Most delay prediction models cheat. They include features like `DEPARTURE_DELAY`, `TAXI_OUT`, or delay cause breakdown columns — information that only exists *after the flight has already left the gate*. That makes accuracy look good on paper while being completely useless in deployment.

This model predicts delay using **only what you know at booking time**: route, airline, scheduled time, date, and distance.

---

## Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.7242 |
| Recall (Delayed flights caught) | 66% |
| Precision (Delayed flags that were correct) | 30% |
| Dataset size | 5,714,008 flights |
| Delay rate (FAA definition: 15+ min late) | 17.9% |

**On 1.14M test flights, the model correctly flagged 135,942 out of 204,700 delayed flights.**

The precision-recall tradeoff is intentional. `scale_pos_weight` tells the model to aggressively flag likely delays rather than miss them — appropriate for a risk-scoring use case.

---

## Why These Numbers Are Honest

The ceiling for scheduling-only delay prediction is around 0.72–0.75 AUC. The information that actually causes delays — live weather, incoming aircraft status, real-time air traffic — is not in this dataset. Routing, airline, and time-of-day encode *historical tendencies*, not *day-of conditions*.

A model that claims 90%+ accuracy on this problem is almost certainly using post-departure features. This one doesn't.

---

## Pipeline

```
Raw CSV (5.7M rows)
    │
    ▼
Step 1 — EDA: missing value audit, column type check
    │
    ▼
Step 2 — Missing Value Handling (order matters)
    │   ├── Remove CANCELLED flights first (never departed — not "on time")
    │   ├── Remove DIVERTED flights (landed at wrong airport — ARRIVAL_DELAY = NaN)
    │   └── Fill remaining ARRIVAL_DELAY NaN with 0 (recording errors only)
    │
    ▼
Step 3 — Leakage Removal
    │   Drop all post-departure columns:
    │   AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY,
    │   LATE_AIRCRAFT_DELAY, WEATHER_DELAY, ARRIVAL_TIME,
    │   WHEELS_ON, TAXI_IN, ELAPSED_TIME, AIR_TIME
    │
    ▼
Step 4 — Target Variable
    │   DELAYED = (ARRIVAL_DELAY > 15).astype(int)
    │   FAA official definition: 15+ minutes late
    │
    ▼
Step 5 — Feature Engineering
    │   HOUR = SCHEDULED_DEPARTURE // 100
    │   IS_WEEKEND, IS_MORNING, IS_NIGHT
    │   ROUTE = ORIGIN_AIRPORT + '_' + DESTINATION_AIRPORT
    │   Label encoding for categorical columns
    │
    ▼
Step 6 — Train/Test Split (stratified, 80/20)
    │
    ▼
Step 7 — Target Encoding (computed on train only → mapped to test)
    │   AIRLINE_DELAY_RATE, ROUTE_DELAY_RATE,
    │   ORIGIN_DELAY_RATE, HOUR_DELAY_RATE
    │
    ▼
Step 8 — XGBoost Training
    │   n_estimators=300, max_depth=6, learning_rate=0.1
    │   scale_pos_weight handles 82/18 class imbalance
    │
    ▼
Step 9 — Evaluation
    Classification report, AUC-ROC, confusion matrix, feature importance
```

---

## Features Used

All features are known **before the flight departs**.

| Feature | Type | Description |
|---|---|---|
| `MONTH` | Temporal | Month of flight |
| `DAY_OF_WEEK` | Temporal | Day of week (1=Mon, 7=Sun) |
| `DAY` | Temporal | Day of month |
| `HOUR` | Engineered | Departure hour extracted from `SCHEDULED_DEPARTURE` |
| `IS_WEEKEND` | Engineered | 1 if Saturday or Sunday |
| `IS_MORNING` | Engineered | 1 if departure before 10 AM |
| `IS_NIGHT` | Engineered | 1 if departure at or after 8 PM |
| `AIRLINE` | Categorical | Encoded airline identifier |
| `ORIGIN_AIRPORT` | Categorical | Encoded origin airport |
| `DESTINATION_AIRPORT` | Categorical | Encoded destination airport |
| `ROUTE` | Engineered | Directional route (ORIGIN_DESTINATION), encoded |
| `SCHEDULED_DEPARTURE` | Scheduling | Raw scheduled departure time |
| `SCHEDULED_ARRIVAL` | Scheduling | Raw scheduled arrival time |
| `SCHEDULED_TIME` | Scheduling | Planned flight duration |
| `DISTANCE` | Route | Distance between airports in miles |
| `AIRLINE_DELAY_RATE` | Target-encoded | Historical delay rate for this airline (train set only) |
| `ROUTE_DELAY_RATE` | Target-encoded | Historical delay rate for this route (train set only) |
| `ORIGIN_DELAY_RATE` | Target-encoded | Historical delay rate for this origin (train set only) |
| `HOUR_DELAY_RATE` | Target-encoded | Historical delay rate for this departure hour (train set only) |

---

## Top Features by Importance

```
ROUTE_DELAY_RATE       0.241   ← strongest signal: some routes are chronically late
HOUR_DELAY_RATE        0.201   ← delays accumulate through the day
SCHEDULED_DEPARTURE    0.145   ← granular time signal
MONTH                  0.068   ← seasonal patterns
DAY                    0.054
AIRLINE_DELAY_RATE     0.051
AIRLINE                0.049
SCHEDULED_ARRIVAL      0.042
DAY_OF_WEEK            0.035
ORIGIN_DELAY_RATE      0.021
```

---

## Key Engineering Decisions

**Why remove cancelled and diverted flights before filling NaN?**
Cancelled flights have `ARRIVAL_DELAY = NaN` but they are not "on time" — they never departed. Filling their NaN with 0 would corrupt the target. Diverted flights landed at the wrong airport, so their delay is undefined. Both are removed first; only then is the remaining NaN (pure recording errors) filled with 0.

**Why is `DEPARTURE_DELAY` excluded?**
`DEPARTURE_DELAY` is only known after the aircraft pushes back from the gate. Including it would make the model accurate in training and useless in production. All post-departure columns are dropped in Step 3.

**Why is target encoding done after the train/test split?**
Target encoding computes delay rates from the `DELAYED` column. If computed on the full dataset, each row's own outcome influences its own feature value — that is data leakage. Computing rates on training data only, then mapping them to the test set, ensures the test set never touches its own labels.

**Why `scale_pos_weight`?**
The dataset is 82% on-time, 18% delayed. Without correction, the model learns to predict "on time" for everything and achieves 82% accuracy while catching zero delays. `scale_pos_weight = 4.58` tells XGBoost that missing a delayed flight costs 4.58× more than a false alarm.

---

## Project Structure

```
flight-delay-predictor/
├── The_Runway.ipynb.ipynb   # Full pipeline — run top to bottom
├── flights.csv.zip                 # Source dataset
├── flight_delay_artifacts.joblib   # Model + encoders + features   
├── flight_delay_model.joblib       # Trained XGBoost model
└── README.md
```

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

Open `flight_delay_prediction.ipynb` in Google Colab and run all cells in order.

---

## Load the Saved Model

```python
import joblib
artifacts = joblib.load('flight_delay_artifacts.joblib')
model    = artifacts['model']
encoders = artifacts['encoders']
features = artifacts['features']
```

---

## Dataset

[US Flights 2015 — Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)

Covers all US domestic flights in 2015. Raw dataset: ~5.8M rows, 31 columns.

---

## What Would Actually Improve This

In order of expected impact:

1. **Live weather data** — temperature, precipitation, wind at origin/destination at departure time. Weather causes ~70% of delays per FAA research. Free historical data available from [Open-Meteo](https://open-meteo.com).
2. **Tail number / previous leg delay** — if the incoming aircraft was already late, propagated delay is near-certain. This is the biggest non-weather predictor used by real airline systems.
3. **Airport congestion** — number of departures from the same airport in the same hour.

These are data availability constraints, not model constraints. No amount of hyperparameter tuning recovers information that isn't in the features.
