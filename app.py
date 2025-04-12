from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to access the backend

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("hospital-resource-manage-firebase-key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Global variables to simulate session state (could also be handled with a persistent data store)
resource_state = {}
patient_queue = []


# ---------- Utility Functions ----------

def load_resource_data():
    df = pd.read_csv("./data/hospital_resources.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    return df


def load_arrival_data():
    df = pd.read_csv("./data/patient_arrivals.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# Initialize resource state
resources_df = load_resource_data()
latest = resources_df.iloc[-1].copy()
total_beds = 100
resource_state = {
    "icu": int(latest["ICU_Available"]),
    "mri": int(latest["MRI_In_Use"]),
    "general": total_beds - int(latest["Bed_Occupancy_Rate"]),
    "ventilators": int(latest["Ventilators_In_Use"])
}


# ---------- API Endpoints ----------

@app.route("/api/resources", methods=["GET"])
def get_resources():
    return jsonify(resource_state)


@app.route("/api/queue", methods=["GET"])
def get_queue():
    # Return current patient queue sorted by urgency (High -> Medium -> Low)
    urgency_rank = {"High": 0, "Medium": 1, "Low": 2}
    sorted_queue = sorted(patient_queue, key=lambda x: urgency_rank[x["urgency"]])
    return jsonify(sorted_queue)


@app.route("/api/patients", methods=["POST"])
def register_patient():
    global resource_state, patient_queue
    data = request.get_json()
    name = data.get("name")
    age = data.get("age")
    problem = data.get("problem")
    urgency = data.get("urgency")
    resource_choice = data.get("resourceChoice")  # For medium urgency patients

    msg = ""
    if urgency == "High":
        # For high urgency patients, allocate ICU, MRI, and Ventilator
        if resource_state["icu"] > 0:
            resource_state["icu"] -= 1
            msg += "✅ ICU allocated. "
        else:
            msg += "⚠ ICU not available. "

        if resource_state["mri"] < 10:
            resource_state["mri"] += 1
            msg += "MRI used. "
        else:
            msg += "⚠ MRI not available. "

        if resource_state["ventilators"] > 0:
            resource_state["ventilators"] -= 1
            msg += "Ventilator allocated. "
        else:
            msg += "⚠ Ventilator not available. "

    elif urgency == "Medium":
        # Resource allocation based on the selected resource choice
        if resource_choice == "ICU":
            if resource_state["icu"] > 0:
                resource_state["icu"] -= 1
                msg = f"✅ ICU bed allocated to {name}."
            else:
                msg = "⚠ ICU not available."
        elif resource_choice == "General":
            if resource_state["general"] > 0:
                resource_state["general"] -= 1
                msg = f"✅ General ward bed allocated to {name}."
            else:
                msg = "⚠ General ward beds not available."
        elif resource_choice == "Ventilator":
            if resource_state["ventilators"] > 0:
                resource_state["ventilators"] -= 1
                msg = f"✅ Ventilator allocated to {name}."
            else:
                msg = "⚠ Ventilator not available."
    else:  # Low urgency
        if resource_state["general"] > 0:
            resource_state["general"] -= 1
            msg = f"✅ General ward bed allocated to {name}."
        else:
            msg = "⚠ No general ward beds available."

    # Create patient record
    patient = {
        "name": name,
        "age": age,
        "problem": problem,
        "urgency": urgency,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Save to Firebase Firestore
    try:
        doc_ref = db.collection("patients").add(patient)
        patient["firestore_id"] = doc_ref[1].id
    except Exception as e:
        return jsonify({"error": f"Failed to save patient to Firebase: {e}"}), 500

    # Append to in-memory queue and sort
    patient_queue.append(patient)
    urgency_rank = {"High": 0, "Medium": 1, "Low": 2}
    patient_queue.sort(key=lambda x: urgency_rank[x["urgency"]])
    return jsonify({"message": f"Patient registered: {msg}", "patient": patient})


@app.route("/api/patients/<string:patient_id>", methods=["DELETE"])
def complete_patient(patient_id):
    global patient_queue
    try:
        # Delete patient from Firebase
        db.collection("patients").document(patient_id).delete()
    except Exception as e:
        return jsonify({"error": f"Failed to delete patient from Firebase: {e}"}), 500

    # Remove patient from in-memory queue
    patient_queue = [p for p in patient_queue if p.get("firestore_id") != patient_id]
    return jsonify({"message": "Patient removed successfully."})


@app.route("/api/reset", methods=["POST"])
def reset_system():
    global resource_state, patient_queue
    # Reset patient collection in Firebase
    try:
        patients = db.collection("patients").stream()
        for doc in patients:
            doc.reference.delete()
    except Exception as e:
        return jsonify({"error": f"Firebase error: {e}"}), 500

    # Reset the in-memory queue and resources state
    patient_queue = []
    latest = load_resource_data().iloc[-1]
    resource_state = {
        "icu": int(latest["ICU_Available"]),
        "mri": int(latest["MRI_In_Use"]),
        "general": total_beds - int(latest["Bed_Occupancy_Rate"]),
        "ventilators": int(latest["Ventilators_In_Use"])
    }
    return jsonify({"message": "System reset successfully."})


@app.route("/api/forecast/occupancy", methods=["GET"])
def arima_forecast():
    # ARIMA: Bed Occupancy Forecast for next 7 days
    ts = load_resource_data().set_index("Date")["Bed_Occupancy_Rate"].resample("D").mean().interpolate()
    try:
        arima_model = ARIMA(ts, order=(3, 1, 2)).fit()
        forecast = arima_model.forecast(steps=7)
        
        # Convert Timestamp keys to string
        forecast_data = {str(date): float(value) for date, value in forecast.to_dict().items()}
    
    except Exception as e:
        return jsonify({"error": f"Forecast error: {e}"}), 500
    
    return jsonify(forecast_data)


@app.route("/api/forecast/beds", methods=["GET"])
def xgb_forecast():
    # XGBoost: Predicted Beds Forecast
    arrivals_df = load_arrival_data()
    arrivals_df["dayofweek"] = pd.to_datetime(arrivals_df["date"]).dt.dayofweek
    arrivals_df["trend"] = np.arange(len(arrivals_df))
    
    # Prepare training data and train model
    X = arrivals_df[["dayofweek", "trend"]]
    y = arrivals_df["patients_queued"]
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Forecast for the next 30 days
    future_days = 30
    future = pd.DataFrame({
        "dayofweek": [(datetime.today() + timedelta(days=i)).weekday() for i in range(future_days)],
        "trend": np.arange(len(arrivals_df), len(arrivals_df) + future_days)
    })
    future["predicted_beds"] = np.round(model.predict(future)).astype(int)
    future["date"] = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(future_days)]
    return jsonify(future.to_dict(orient="records"))


if __name__ == '__main__':
    app.run(debug=True,port=5001)