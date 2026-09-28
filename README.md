# ⚡ Smart Building Energy Cost Predictor

A simple Streamlit web app that predicts how much energy a building will consume, what it'll cost, and how much CO₂ it'll produce — based on a few basic details like building type, area, occupants, and temperature.

I built this to explore how machine learning can help estimate energy usage for buildings before the bills actually come in, which could be useful for facility managers, homeowners, or anyone trying to plan ahead.

🔗 **Live app:** [smart-building-energy-cost-predictor](https://smart-building-energy-predictorr.streamlit.app/)

---

## What it does

You enter a few details about a building:
- Building type (Residential / Commercial / Industrial)
- Area (in sq ft)
- Number of occupants
- Number of appliances in use
- Average temperature
- Whether it's a weekday or weekend

The app runs these through a pre-trained regression model and gives you back:
- **Predicted energy consumption** (in kWh)
- **Estimated cost** (in ₹, based on a rate of ₹8/kWh)
- **Estimated CO₂ emissions** (based on 0.82 kg CO₂ per kWh)

## Tech stack

- **Streamlit** – for the web interface
- **scikit-learn** – model training and the `StandardScaler` used to preprocess inputs
- **pandas** – data handling
- **pickle** – for loading the trained model

## How it works

The model was trained on a dataset (`data/raw_data.csv`) containing building type, square footage, occupants, appliances, temperature, day of week, and actual energy consumption. Categorical columns (building type, day of week) are one-hot encoded, and numerical features are scaled with `StandardScaler` before being fed into the model.

At runtime, the app rebuilds the same feature set from your inputs, scales it using the same logic, and passes it to the saved model (`models/cost_prediction.pkl`) to get a prediction.


## Project structure

```
├── app.py               # Main Streamlit app
├── requirements.txt      # Python dependencies
├── models/
│   └── cost_prediction.pkl   # Trained regression model
└── data/
    └── raw_data.csv       # Training data (used to rebuild the scaler)
```

thanks!!!!!!!