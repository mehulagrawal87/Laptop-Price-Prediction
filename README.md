# 💻 Laptop Price Predictor

An end-to-end Machine Learning application that predicts the estimated market price of a laptop based on its hardware, display, storage, and system specifications.

The project combines Machine Learning with an interactive Streamlit web application to provide price estimation along with laptop performance, portability, appraisal, and market-position insights.

## 🚀 Live Demo

🌐 **Try the application:**  
https://laptop-price-predictor99.streamlit.app/

---

## 🚀 Features

- Laptop price prediction using Machine Learning
- Interactive Streamlit web application
- Stacking Regression model
- Random Forest, Gradient Boosting, XGBoost and Ridge Regression
- Automatic categorical feature encoding
- Hardware-based performance scoring
- Estimated market value and appraisal range
- Market-position percentile
- Laptop profile classification
- Interactive visualizations using Plotly
- Modern and responsive user interface

---

## 🧠 Machine Learning Model

The project uses a **Stacking Regressor** consisting of multiple base models:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Ridge Regression as the final estimator

Categorical features are processed using `OneHotEncoder`, while the complete workflow is implemented using a Scikit-learn `Pipeline`.

The target price is transformed using a logarithmic transformation during training and converted back to the original price scale during prediction.

---

## 📊 Input Features

The application allows users to configure a laptop using features such as:

### Brand & System
- Laptop Brand
- Laptop Type / Form Factor
- Operating System

### Hardware
- CPU
- GPU
- RAM
- SSD
- HDD

### Display & Portability
- Screen Size
- Screen Resolution
- Weight
- Touchscreen
- IPS Display

The application also calculates PPI from screen resolution and screen size before sending the input to the trained model.

---

## 📈 Prediction Output

After entering the laptop specifications, the application generates:

- **Estimated Market Value**
- **Appraisal Range**
- **Laptop Profile**
- **Market Position Percentile**
- **Hardware Summary**
- **Performance Insights**

The application provides an estimated price range around the predicted market value.

---

## ⚡ Laptop Profile Classification

Based on the selected specifications, the application categorizes laptops into different profiles, including:

- Extreme Gaming / Creator Beast
- Professional Powerhouse Workstation
- Sleek Ultra-Premium Portable
- Balanced Modern Office / Developer Machine
- Standard Entry-Level Companion

These profiles are determined using factors such as laptop type, GPU, RAM, CPU, and weight.

---

## 📊 Performance & Portability Insights

The application provides additional insights based on laptop specifications, including:

- CPU Performance
- GPU Capability
- RAM Capacity
- SSD Availability
- Portability

These scores help users understand the practical characteristics of a laptop beyond its predicted price.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning & preprocessing |
| XGBoost | Gradient boosting |
| Streamlit | Web application |
| Plotly | Interactive visualizations |
| Pickle | Model serialization |

---

## 📁 Project Structure

```text
Laptop-Price-Prediction/
│
├── app.py
├── main.py
├── Laptop-Price-Predictor.ipynb
├── laptop_data.csv
├── df.pkl
├── pipe.pkl
├── requirements.txt
├── README.md
└── .gitignore
