from flask import Flask, jsonify, render_template
import requests
import sqlite3
import numpy as np
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv
import os

# create Flask app
app = Flask(__name__)

# OpenWeatherMap API 
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_URL = 'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric'

# Open-Elevation API URL
ELEVATION_API_URL = 'https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}'

# sqlite database
def init_db():
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            pressure REAL,
            wind_speed REAL,
            humidity REAL,
            altitude REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# call this function when starting the app
init_db()

# generate random sensor data for training
def generateTrainingData():
    sensorData = np.random.uniform(low=[39, 950, 0, 60, 0], high=[94, 1050, 15, 100, 345], size=[1000, 5]) # temp, pressure, wind speed, humidity, altitude
    return sensorData

# train the isolation forest model
def trainAnomalyModel():
    trainingData = generateTrainingData()
    model = IsolationForest(contamination=0.01)  # lowered to 1% anomaly rate
    model.fit(trainingData)
    return model

# initialize the anomaly detection model
anomalyModel = trainAnomalyModel()

# fetch real sensor data using Open-Elevation API
def fetchAltitude(lat, lon):
    response = requests.get(ELEVATION_API_URL.format(lat=lat, lon=lon))
    if response.status_code == 200:  # Check if request is successful
        elevationData = response.json()
        if 'results' in elevationData:
            altitude = elevationData['results'][0]['elevation']
            return altitude
    return -1  # return default if API fails

# fetch real sensor data using One Call API 3.0
def fetchRealData(lat=28.53, lon=-81.37): # orlando's lat and lon
    response = requests.get(API_URL.format(lat=lat, lon=lon, api_key=API_KEY))
    if response.status_code == 200:  # Check if request is successful
        data = response.json()
        # fetch altitude from Open-Elevation API
        altitude = fetchAltitude(lat, lon)
        # extract relevant data
        sensorData = {
            'temperature': data['current']['temp'],
            'pressure': data['current']['pressure'],
            'wind_speed': data['current']['wind_speed'],
            'humidity': data['current']['humidity'],
            'altitude': altitude 
        }
        return sensorData
    else:
        print(f"Error: OpenWeatherMap API request failed with status code {response.status_code}")
        return None  # Return None if API fails

# store the data
def store_sensor_data(data):
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO sensor_data (temperature, pressure, wind_speed, humidity, altitude)
        VALUES (?, ?, ?, ?, ?)
    ''', (data['temperature'], data['pressure'], data['wind_speed'], data['humidity'], data['altitude']))
    conn.commit()
    conn.close()

# check if the sensor data is an anomaly
def checkAnomaly(newData):
    newDataArray = np.array([list(newData.values())]).reshape(1, -1)
    anomaly = anomalyModel.predict(newDataArray)
    return anomaly[0] == -1

# define a route to serve sensor data
@app.route('/sensorData')
def sensor_data():
    data = fetchRealData()  # Fetch real-time data
    if data:
        store_sensor_data(data)  # Store data in the database
        data['anomaly'] = int(checkAnomaly(data))  # Anomaly detection
        return jsonify(data)
    else:
        return jsonify({"error": "Failed to fetch sensor data"}), 500

# define a route to historical data
@app.route('/historicalData')
def historical_data():
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('SELECT temperature, pressure, wind_speed, humidity, altitude, timestamp FROM sensor_data ORDER BY timestamp DESC LIMIT 100')
    data = c.fetchall()
    conn.close()

    historical_data = []
    for row in data:
        historical_data.append({
            'temperature': row[0],
            'pressure': row[1],
            'wind_speed': row[2],
            'humidity': row[3],
            'altitude': row[4],
            'timestamp': row[5]
        })

    return jsonify(historical_data)

# define a route to template
@app.route('/')
def index():
    return render_template('index.html')

# run app
if __name__ == '__main__':
    app.run(debug=True)
