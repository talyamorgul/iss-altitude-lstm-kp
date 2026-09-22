# ISS Altitude Prediction from Geomagnetic Kp Index (LSTM)

Code and data for "Persistence, Not Prediction: Evaluating an LSTM 
Against a Naive Baseline for ISS Altitude Forecasting from 
Geomagnetic Activity."

## Contents
- `absolute_altitude_model.py` — original LSTM predicting next-day 
  absolute ISS altitude from a 10-day Kp window
- `lstm_delta_model.py` — reformulated LSTM predicting day-over-day 
  altitude change, reconstructed against the true previous-day value
- `iss_25544_2023.csv` — ISS TLE history for 2023 (Space-Track.org, 
  NORAD ID 25544, gp_history)
- `LSTM_Physical_Results_with_Reboost.csv`, `LSTM_Delta_Results.csv` 
  — predicted vs. actual results underlying Table 2 of the manuscript

## Data sources
- Geomagnetic Kp index: NASA DONKI (https://api.nasa.gov/DONKI/GST)
- ISS orbital elements: Space-Track.org (https://www.space-track.org)

## Requirements
requests, pandas, numpy, scikit-learn, tensorflow
