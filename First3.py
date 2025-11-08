import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, Topos
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
import sqlite3
from pathlib import Path
import os

# ===================== CONSTANTS =====================
MIN_ISS_ALTITUDE = 10.0  # Minimum ISS elevation threshold (°)
SUN_ALTITUDE_THRESHOLD = -6.0  # Nighttime threshold (°)
PREDICTION_DAYS = 30  # Prediction length in days
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"
DB_NAME = "iss_observations.db"


# ===================== DATABASE FUNCTIONS =====================
def init_db():
    """Initialize SQLite database for storing predictions and observations"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  latitude REAL,
                  longitude REAL,
                  prediction_date TEXT,
                  rise_time_local TEXT,
                  set_time_local TEXT,
                  peak_time_local TEXT,
                  min_sun_alt REAL,
                  iss_peak_alt REAL,
                  duration_min REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Create observations table - 修正了注释格式，去掉了#
    c.execute('''CREATE TABLE IF NOT EXISTS observations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  prediction_id INTEGER,
                  observed BOOLEAN,
                  actual_rise_time TEXT,
                  actual_peak_time TEXT,
                  actual_set_time TEXT,
                  visibility_rating INTEGER,  -- 1-5 scale
                  notes TEXT,
                  observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(prediction_id) REFERENCES predictions(id))''')

    conn.commit()
    conn.close()


def save_predictions_to_db(predictions, lat, lon):
    """Save predictions to database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    prediction_ids = []
    for pred in predictions:
        c.execute('''INSERT INTO predictions 
                     (latitude, longitude, prediction_date, rise_time_local, 
                      set_time_local, peak_time_local, min_sun_alt, 
                      iss_peak_alt, duration_min)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (lat, lon, pred["Date"], pred["Rise Time (Local)"],
                   pred["Set Time (Local)"], pred["Peak Time (Local)"],
                   pred["Min Sun Alt(°)"], pred["ISS Peak Alt(°)"],
                   pred["Duration (min)"]))
        prediction_ids.append(c.lastrowid)

    conn.commit()
    conn.close()
    return prediction_ids


def save_observation_to_db(prediction_id, observed, actual_times, rating, notes):
    """Save observation data to database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute('''INSERT INTO observations 
                 (prediction_id, observed, actual_rise_time, 
                  actual_peak_time, actual_set_time, visibility_rating, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (prediction_id, 1 if observed else 0,
               actual_times.get('rise', ''), actual_times.get('peak', ''),
               actual_times.get('set', ''), rating, notes))

    conn.commit()
    conn.close()


def get_all_predictions_with_observations():
    """Get all predictions with their associated observations"""
    conn = sqlite3.connect(DB_NAME)
    query = '''SELECT p.*, o.observed, o.actual_rise_time, o.actual_peak_time,
                      o.actual_set_time, o.visibility_rating
               FROM predictions p
               LEFT JOIN observations o ON p.id = o.prediction_id'''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ===================== UTILITY FUNCTIONS =====================
def get_timezone(lat: float, lon: float) -> ZoneInfo:
    """Return timezone based on latitude/longitude"""
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    return ZoneInfo(tz_name) if tz_name else ZoneInfo("UTC")


def utc_to_local(utc_dt: datetime, local_tz: ZoneInfo) -> datetime:
    """Convert UTC time to local timezone"""
    return utc_dt.astimezone(local_tz)


def load_iss_tle() -> tuple:
    """Load ISS TLE data (with exception handling)"""
    stations_url = 'https://celestrak.org/NORAD/elements/stations.txt'
    try:
        satellites = load.tle_file(stations_url)
        iss = next((sat for sat in satellites if 'ISS' in sat.name), None)
        if not iss:
            raise ValueError("ISS not found in TLE data")
        epoch = iss.epoch.utc_strftime(DATE_FORMAT + " " + TIME_FORMAT)
        return iss, epoch
    except Exception as e:
        st.error(f"Failed to load TLE data: {e}")
        return None, None


# ===================== PREDICTION FUNCTIONS =====================
def generate_iss_predictions(lat, lon, start_date):
    """Generate ISS visibility predictions"""
    # Timezone handling
    local_tz = get_timezone(lat, lon)
    local_start = datetime.strptime(start_date, DATE_FORMAT).replace(
        hour=0, minute=0, second=0, tzinfo=local_tz)
    utc_start = local_start.astimezone(ZoneInfo("UTC"))

    # Load ISS data
    iss, epoch = load_iss_tle()
    if not iss:
        return None, None

    # Initialize objects
    ts = load.timescale()
    observer = Topos(latitude_degrees=lat, longitude_degrees=lon, elevation_m=50)
    eph = load('de421.bsp')
    earth = eph['earth']
    sun = eph['sun']
    observer_earth = earth + observer

    # Transit event calculation
    passes_data = []

    # Time range
    t0 = ts.utc(utc_start.year, utc_start.month, utc_start.day)
    t1 = t0 + timedelta(days=PREDICTION_DAYS)

    # Get events: rise / culmination / set
    t_events, event_types = iss.find_events(observer, t0, t1, altitude_degrees=MIN_ISS_ALTITUDE)

    # Extract rise/set pairs
    rise_set_pairs = []
    current_rise = None
    for t, event in zip(t_events, event_types):
        if event == 0:  # rise
            current_rise = t
        elif event == 2 and current_rise is not None:  # set
            rise_set_pairs.append((current_rise, t))
            current_rise = None

    # Process each pass
    for t_rise, t_set in rise_set_pairs:
        dt_rise = t_rise.utc_datetime()
        dt_set = t_set.utc_datetime()
        dt_peak = dt_rise + (dt_set - dt_rise) / 2
        t_peak = ts.from_datetime(dt_peak)

        sun_alt_rise, _, _ = observer_earth.at(t_rise).observe(sun).apparent().altaz()
        sun_alt_peak, _, _ = observer_earth.at(t_peak).observe(sun).apparent().altaz()
        sun_alt_set, _, _ = observer_earth.at(t_set).observe(sun).apparent().altaz()

        min_sun_alt = min(sun_alt_rise.degrees, sun_alt_peak.degrees, sun_alt_set.degrees)

        iss_sunlit = (iss.at(t_rise).is_sunlit(eph) or
                      iss.at(t_peak).is_sunlit(eph) or
                      iss.at(t_set).is_sunlit(eph))

        iss_position = iss.at(t_peak)
        observer_position = observer.at(t_peak)
        alt, az, _ = (iss_position - observer_position).altaz()
        iss_peak_alt = alt.degrees

        if (sun_alt_rise.degrees < SUN_ALTITUDE_THRESHOLD or
            sun_alt_peak.degrees < SUN_ALTITUDE_THRESHOLD or
            sun_alt_set.degrees < SUN_ALTITUDE_THRESHOLD) and iss_sunlit:
            local_rise = utc_to_local(dt_rise, local_tz)
            local_set = utc_to_local(dt_set, local_tz)
            local_peak = utc_to_local(dt_peak, local_tz)

            passes_data.append({
                "Date": local_rise.strftime(DATE_FORMAT),
                "Rise Time (Local)": local_rise.strftime(TIME_FORMAT),
                "Set Time (Local)": local_set.strftime(TIME_FORMAT),
                "Peak Time (Local)": local_peak.strftime(TIME_FORMAT),
                "Min Sun Alt(°)": round(min_sun_alt, 1),
                "ISS Peak Alt(°)": round(iss_peak_alt, 1),
                "Duration (min)": round((dt_set - dt_rise).total_seconds() / 60, 1)
            })

    return passes_data, epoch


# ===================== ANALYSIS & VISUALIZATION =====================
def calculate_accuracy_metrics(df):
    """Calculate prediction accuracy metrics"""
    # Filter to only predictions that have observations
    observed_predictions = df[df['observed'].notna()]

    if len(observed_predictions) == 0:
        return None

    # Calculate overall accuracy (predicted pass was actually observed)
    accuracy = observed_predictions['observed'].mean() * 100

    # Calculate time differences where we have actual times
    time_diff_metrics = {}

    # Process rise time differences
    rise_times = observed_predictions[observed_predictions['actual_rise_time'] != '']
    if len(rise_times) > 0:
        rise_diff = []
        for _, row in rise_times.iterrows():
            pred = datetime.strptime(f"{row['prediction_date']} {row['rise_time_local']}",
                                     f"{DATE_FORMAT} {TIME_FORMAT}")
            actual = datetime.strptime(f"{row['prediction_date']} {row['actual_rise_time']}",
                                       f"{DATE_FORMAT} {TIME_FORMAT}")
            rise_diff.append(abs((actual - pred).total_seconds() / 60))  # minutes
        time_diff_metrics['rise'] = {
            'mean': np.mean(rise_diff),
            'median': np.median(rise_diff),
            'max': np.max(rise_diff)
        }

    # Process peak time differences
    peak_times = observed_predictions[observed_predictions['actual_peak_time'] != '']
    if len(peak_times) > 0:
        peak_diff = []
        for _, row in peak_times.iterrows():
            pred = datetime.strptime(f"{row['prediction_date']} {row['peak_time_local']}",
                                     f"{DATE_FORMAT} {TIME_FORMAT}")
            actual = datetime.strptime(f"{row['prediction_date']} {row['actual_peak_time']}",
                                       f"{DATE_FORMAT} {TIME_FORMAT}")
            peak_diff.append(abs((actual - pred).total_seconds() / 60))  # minutes
        time_diff_metrics['peak'] = {
            'mean': np.mean(peak_diff),
            'median': np.median(peak_diff),
            'max': np.max(peak_diff)
        }

    return {
        'total_predictions': len(df),
        'total_observed': len(observed_predictions),
        'accuracy': accuracy,
        'time_diffs': time_diff_metrics
    }


def create_visualizations(df):
    """Create visualization of prediction accuracy"""
    # Filter to only predictions that have observations
    observed_predictions = df[df['observed'].notna()]

    if len(observed_predictions) == 0:
        st.info("No observation data available for visualization. Enter some observation data first.")
        return

    # 1. Observation Rate Bar Chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    observation_counts = observed_predictions['observed'].value_counts()
    ax1.bar(['Observed', 'Not Observed'],
            [observation_counts.get(1, 0), observation_counts.get(0, 0)],
            color=['green', 'red'])
    ax1.set_title('Observation Rate of Predicted Passes')
    ax1.set_ylabel('Number of Passes')
    ax1.set_ylim(0, max(observation_counts) + 1)
    st.pyplot(fig1)

    # 2. Time Difference Histogram
    time_diffs = []
    labels = []

    # Check for rise time differences
    rise_times = observed_predictions[observed_predictions['actual_rise_time'] != '']
    if len(rise_times) > 0:
        for _, row in rise_times.iterrows():
            pred = datetime.strptime(f"{row['prediction_date']} {row['rise_time_local']}",
                                     f"{DATE_FORMAT} {TIME_FORMAT}")
            actual = datetime.strptime(f"{row['prediction_date']} {row['actual_rise_time']}",
                                       f"{DATE_FORMAT} {TIME_FORMAT}")
            time_diffs.append(abs((actual - pred).total_seconds() / 60))  # minutes
            labels.append('Rise')

    # Check for peak time differences
    peak_times = observed_predictions[observed_predictions['actual_peak_time'] != '']
    if len(peak_times) > 0:
        for _, row in peak_times.iterrows():
            pred = datetime.strptime(f"{row['prediction_date']} {row['peak_time_local']}",
                                     f"{DATE_FORMAT} {TIME_FORMAT}")
            actual = datetime.strptime(f"{row['prediction_date']} {row['actual_peak_time']}",
                                       f"{DATE_FORMAT} {TIME_FORMAT}")
            time_diffs.append(abs((actual - pred).total_seconds() / 60))  # minutes
            labels.append('Peak')

    if time_diffs:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(time_diffs, bins=10, alpha=0.7)
        ax2.set_title('Distribution of Time Prediction Errors (minutes)')
        ax2.set_xlabel('Time Difference (minutes)')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)

    # 3. Visibility vs Predicted Altitude
    rated_obs = observed_predictions[observed_predictions['visibility_rating'].notna()]
    if len(rated_obs) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(rated_obs['iss_peak_alt'], rated_obs['visibility_rating'])
        ax3.set_title('Visibility Rating vs Predicted ISS Peak Altitude')
        ax3.set_xlabel('Predicted ISS Peak Altitude (°)')
        ax3.set_ylabel('Visibility Rating (1-5)')
        ax3.set_ylim(0, 6)
        st.pyplot(fig3)


# ===================== STREAMLIT APP =====================
def main():
    st.set_page_config(page_title="ISS Visibility Predictor", layout="wide")
    st.title("ISS Visibility Predictor & Tracker")

    # Initialize database
    init_db()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Generate Predictions", "Log Observations", "Analysis & Visualization"])

    with tab1:
        st.header("Generate ISS Visibility Predictions")

        # User input
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=0.0, step=0.01)
            lon = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01)

        with col2:
            today = datetime.now().strftime(DATE_FORMAT)
            start_date = st.text_input("Start Date (YYYY-MM-DD)", value=today)

        if st.button("Generate Predictions"):
            # Validate inputs
            try:
                datetime.strptime(start_date, DATE_FORMAT)
            except ValueError:
                st.error("Invalid date format. Please use YYYY-MM-DD.")
                return

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                st.error("Latitude must be between -90 and 90, longitude between -180 and 180.")
                return

            # Generate predictions
            with st.spinner("Calculating ISS visibility predictions..."):
                predictions, epoch = generate_iss_predictions(lat, lon, start_date)

                if predictions:
                    st.success(f"Successfully generated {len(predictions)} predictions! TLE Epoch: {epoch}")

                    # Save to database
                    prediction_ids = save_predictions_to_db(predictions, lat, lon)

                    # Display predictions
                    df = pd.DataFrame(predictions)
                    st.dataframe(df)

                    # Add download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv,
                        file_name="iss_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No visible nighttime ISS passes found for the given parameters.")

    with tab2:
        st.header("Log Observation Data")

        # Get all predictions without observations
        conn = sqlite3.connect(DB_NAME)
        query = '''SELECT p.id, p.prediction_date, p.rise_time_local, p.peak_time_local, 
                          p.set_time_local, p.iss_peak_alt, p.latitude, p.longitude
                   FROM predictions p
                   LEFT JOIN observations o ON p.id = o.prediction_id
                   WHERE o.id IS NULL'''
        predictions_df = pd.read_sql_query(query, conn)
        conn.close()

        if len(predictions_df) == 0:
            st.info("No predictions available to log observations for. Generate some predictions first.")
        else:
            # Let user select a prediction to log
            prediction_id = st.selectbox(
                "Select a prediction to log an observation for:",
                predictions_df['id'],
                format_func=lambda
                    x: f"ID {x}: {predictions_df.loc[predictions_df['id'] == x, 'prediction_date'].iloc[0]} "
                       f"({predictions_df.loc[predictions_df['id'] == x, 'rise_time_local'].iloc[0]} - "
                       f"{predictions_df.loc[predictions_df['id'] == x, 'set_time_local'].iloc[0]})"
            )

            # Get selected prediction details
            selected_pred = predictions_df[predictions_df['id'] == prediction_id].iloc[0]

            # Display prediction details
            st.subheader("Prediction Details")
            st.write(f"Date: {selected_pred['prediction_date']}")
            st.write(f"Location: Lat {selected_pred['latitude']}, Lon {selected_pred['longitude']}")
            st.write(f"Predicted Rise Time: {selected_pred['rise_time_local']}")
            st.write(f"Predicted Peak Time: {selected_pred['peak_time_local']}")
            st.write(f"Predicted Set Time: {selected_pred['set_time_local']}")
            st.write(f"Predicted Peak Altitude: {selected_pred['iss_peak_alt']}°")

            # Observation form
            st.subheader("Observation Data")
            observed = st.radio("Did you observe the ISS during this pass?", ["Yes", "No"])
            observed_bool = observed == "Yes"

            actual_times = {}
            if observed_bool:
                col1, col2, col3 = st.columns(3)
                with col1:
                    actual_rise = st.text_input("Actual Rise Time (HH:MM:SS)", value=selected_pred['rise_time_local'])
                    actual_times['rise'] = actual_rise
                with col2:
                    actual_peak = st.text_input("Actual Peak Time (HH:MM:SS)", value=selected_pred['peak_time_local'])
                    actual_times['peak'] = actual_peak
                with col3:
                    actual_set = st.text_input("Actual Set Time (HH:MM:SS)", value=selected_pred['set_time_local'])
                    actual_times['set'] = actual_set

                visibility = st.slider("Visibility Rating (1=Poor, 5=Excellent)", 1, 5, 3)
            else:
                visibility = None

            notes = st.text_area("Additional Notes (e.g., weather conditions, obstacles)", "")

            if st.button("Save Observation"):
                # Validate time formats if provided
                time_format_valid = True
                if observed_bool:
                    for time_type, time_str in actual_times.items():
                        try:
                            datetime.strptime(time_str, TIME_FORMAT)
                        except ValueError:
                            st.error(f"Invalid {time_type} time format. Please use HH:MM:SS.")
                            time_format_valid = False
                            break

                if time_format_valid:
                    save_observation_to_db(prediction_id, observed_bool, actual_times, visibility, notes)
                    st.success("Observation saved successfully!")
                    st.rerun()

    with tab3:
        st.header("Prediction Accuracy Analysis")

        # Get all data
        df = get_all_predictions_with_observations()

        if len(df) == 0:
            st.info("No prediction data available. Generate some predictions first.")
        else:
            # Calculate metrics
            metrics = calculate_accuracy_metrics(df)

            if metrics:
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", metrics['total_predictions'])
                with col2:
                    st.metric("Predictions with Observations", metrics['total_observed'])
                with col3:
                    st.metric("Observation Accuracy", f"{metrics['accuracy']:.1f}%")

                # Display time difference metrics
                if metrics['time_diffs']:
                    st.subheader("Time Prediction Accuracy (minutes)")
                    time_metrics_df = pd.DataFrame()

                    for time_type, stats in metrics['time_diffs'].items():
                        time_metrics_df = pd.concat([time_metrics_df, pd.DataFrame({
                            'Time Type': [time_type.capitalize()],
                            'Mean Difference': [f"{stats['mean']:.1f}"],
                            'Median Difference': [f"{stats['median']:.1f}"],
                            'Max Difference': [f"{stats['max']:.1f}"]
                        })], ignore_index=True)

                    st.dataframe(time_metrics_df)

            # Display visualizations
            st.subheader("Accuracy Visualizations")
            create_visualizations(df)

            # Display raw data
            with st.expander("View Raw Data"):
                st.dataframe(df)


if __name__ == "__main__":
    main()