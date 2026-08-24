Forecasts US measles case counts and classifies outbreak risk (Low/Medium/High) for the next 4 weeks, using real CDC and JHU public health data.

Live demo: https://measles-outbreak-predictor-production.up.railway.app/

What it does:
XGBoost regression forecasts total cases over the next 4 weeks
Random Forest classifier predicts outbreak risk with calibrated probabilities per class (e.g. Low: 5%, Medium: 20%, High: 75%)
Served via a Flask dashboard with live predictions and historical trends

Data:
CDC historical annual totals (2010–2024) distributed into weekly estimates, combined with live weekly data from the JHU CSSE Measles Data Repository (2025–present, updated Fridays).
Model performance (5-fold time-series CV)

Features
Lag features (1/2/4/8 weeks), rolling mean/std/max, trend ratio, cyclical week-of-year encoding, and vaccination rate.

API:
GET /api/forecast — current 4-week forecast + risk classification
GET /api/historical — monthly case trend data
GET /api/summary — quick summary stats

Tech stack:
Python, Flask, scikit-learn, XGBoost, pandas, gunicorn — deployed on Railway.
