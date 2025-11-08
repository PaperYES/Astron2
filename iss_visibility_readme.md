# ISS Visibility Predictor & Tracker

A Streamlit-based web application for predicting the visibility of the International Space Station (ISS), logging actual observations, and analyzing prediction accuracy. The app uses astronomical calculations to provide nighttime ISS pass predictions and allows users to track real-world observations.

---

## Features

1. **Generate Predictions**
   - Input latitude, longitude, and start date.
   - Predict ISS passes for the next 30 days.
   - Only nighttime passes where the ISS is illuminated are shown.
   - Display rise, peak, and set times (local timezone), peak altitude, and duration.
   - Save predictions to a local SQLite database.
   - Export predictions as CSV.

2. **Log Observations**
   - Select a prediction from the database.
   - Record whether the ISS was observed.
   - Log actual rise, peak, and set times.
   - Rate visibility on a 1-5 scale.
   - Add notes about weather or obstacles.
   - Save observation data to the database.

3. **Analysis & Visualization**
   - Calculate prediction accuracy and time differences.
   - Visualize observation rates, time prediction errors, and visibility vs predicted peak altitude.
   - View raw prediction and observation data.

---

## Installation

1. **Clone the repository**

```bash
git clone <repository_url>
cd <repository_folder>
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> **Dependencies include:**
>
> - streamlit
> - pandas
> - numpy
> - matplotlib
> - skyfield
> - timezonefinder

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. The app will open in your default browser.

3. Navigate through the tabs:
   - **Generate Predictions**: Input location and date, generate predictions.
   - **Log Observations**: Record real observations for predicted passes.
   - **Analysis & Visualization**: View accuracy metrics and charts.

---

## Database

The app uses a local SQLite database (`iss_observations.db`) with two main tables:

1. **predictions**
   - Stores all ISS visibility predictions.
   - Columns: latitude, longitude, prediction_date, rise/set/peak times, peak altitude, duration, etc.

2. **observations**
   - Stores user observations.
   - Linked to `predictions` via `prediction_id`.
   - Columns: observed (boolean), actual rise/peak/set times, visibility rating, notes, timestamp.

---

## Notes

- Predictions are based on the latest ISS TLE data from [Celestrak](https://celestrak.com/NORAD/elements/stations.txt).
- Only nighttime passes where the ISS is illuminated are considered.
- Local timezone conversion is automatic based on latitude/longitude.
- All user data is stored locally in the SQLite database.

---


## License

This project is licensed under the MIT License.

