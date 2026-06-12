# 🌸 Iris Flower Predictor

A Machine Learning web application that predicts the species of an Iris flower using a **Random Forest Classifier**. Users can enter flower measurements and instantly receive a prediction.

## 📌 Overview

This project classifies iris flowers into one of three species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

The prediction is based on four input features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

## 🚀 Features

* Interactive and user-friendly interface
* Instant flower species prediction
* Random Forest-based classification model
* Responsive design
* Fast and accurate predictions

## 🛠️ Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML
* CSS

## 🤖 Machine Learning Model

The model was trained using the **Random Forest Classifier**, an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

## 📊 Dataset

The project uses the Iris dataset, one of the most widely used datasets for machine learning classification tasks. It contains measurements of iris flowers belonging to three different species.

## ⚙️ How It Works

1. The user enters the flower measurements.
2. The input data is passed to the trained Random Forest model.
3. The model predicts the most likely iris species.
4. The prediction is displayed on the webpage.

## 📂 Project Structure

```text
iris-flower-predictor/
│
├── app.py
├── model.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md
```

## ▶️ Running the Project

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Flask application:

```bash
python app.py
```

3. Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

## 🎯 Result

The application predicts whether the flower belongs to **Setosa**, **Versicolor**, or **Virginica** based on the measurements provided by the user.
