<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Random%20Forest-Regressor-blueviolet?style=for-the-badge"/>

# 🛵 Delivery Time Predictor

**An AI-powered web app that predicts food delivery time using a trained Random Forest model — with a neon-chamki UI built in Streamlit.**

[🚀 Run the App](#-getting-started) · [📊 Model Details](#-model-details) · [🎨 UI Preview](#-ui-preview) · [🤝 Contributing](#-contributing)

---

</div>

## 📌 Overview

This project provides a sleek, interactive **Streamlit dashboard** for predicting delivery time (in minutes) based on real-world delivery factors such as distance, weather, traffic, and courier experience. The prediction engine is a trained **Random Forest Regressor** with 100 estimators and 18 input features.

---

## ✨ Features

- 🎯 **Instant predictions** — enter delivery conditions and get ETA in real time
- 🌦️ **Multi-condition support** — weather, traffic level, time of day, vehicle type
- 📊 **Feature importance viewer** — see which factors influence delivery time most
- 🏍️ **Courier profiling** — accounts for courier experience and vehicle type
- 💅 **Neon chamki UI** — vibrant dark-mode interface with glowing cards and animated result box

---

## 🖼️ UI Preview

| Input Panel | Prediction Result |
|------------|------------------|
| Sliders for distance, prep time, courier experience | Glowing result card with ETA in minutes |
| Dropdowns for weather, traffic, time of day, vehicle | Speed badge: 🚀 Lightning Fast / ⚡ On Time / ⚠️ Expect a Wait |

---

## 📁 Project Structure

```
random-forest/
│
├── app.py              # Streamlit UI application
├── rf_model.pkl        # Trained Random Forest Regressor model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Estimators | 100 trees |
| Input Features | 18 |
| Target Variable | Delivery Time (minutes) |
| Framework | scikit-learn 1.6.1 |

### 📐 Input Features

| Category | Features |
|---|---|
| **Numeric** | `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs` |
| **Weather** | `Weather_Clear`, `Weather_Foggy`, `Weather_Rainy`, `Weather_Snowy`, `Weather_Windy` |
| **Traffic** | `Traffic_Level_High`, `Traffic_Level_Low`, `Traffic_Level_Medium` |
| **Time of Day** | `Time_of_Day_Morning`, `Time_of_Day_Afternoon`, `Time_of_Day_Evening`, `Time_of_Day_Night` |
| **Vehicle** | `Vehicle_Type_Bike`, `Vehicle_Type_Car`, `Vehicle_Type_Scooter` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/delivery-time-predictor.git
cd delivery-time-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

> ⚠️ **Important:** Always use `streamlit run app.py` — never `python app.py`.

The app will open automatically at **http://localhost:8501**

---

## 📦 Requirements

Create a `requirements.txt` with:

```
streamlit
scikit-learn==1.6.1
joblib
pandas
numpy
```

---

## 🎮 How to Use

1. **Route Details** — Set the delivery distance (km) and food preparation time (min)
2. **Courier & Vehicle** — Adjust courier experience and select vehicle type
3. **Conditions** — Choose weather, traffic level, and time of day
4. **Hit Predict** — Click ⚡ PREDICT DELIVERY TIME to get the estimated ETA
5. **Explore** — Expand the feature importance section to understand the model

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Open an issue for bugs
- 💡 Suggest new features via GitHub Issues
- 🔀 Submit a pull request

### Steps to contribute

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request on GitHub
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and lots of ☕

⭐ **Star this repo** if you found it useful!

</div>
