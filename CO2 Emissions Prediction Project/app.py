from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained files
pipeline = joblib.load("co2_pipeline.pkl")
model_frequency = joblib.load("model_frequency.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict():

    data = request.get_json()
    # print(data)

    input_df = pd.DataFrame([{
    "Make": data["make"],
    "Model": data["model"],
    "Vehicle Class": data["vclass"],
    "Engine Size(L)": data["engine"],
    "Cylinders": data["cyl"],
    "Transmission": data["trans"],
    "Fuel Type": data["fuel"],
    "Fuel_City": data["city"],
    "Fuel_Highway": data["hwy"],
    "Fuel_Combine(L/100 km)": data["comb"],
    "Fuel_Combine(mpg)": 282.481 / data["comb"]
    }])

    input_df["Model"] = input_df["Model"].map(model_frequency)
    input_df["Model"] = input_df["Model"].fillna(0)

    prediction = pipeline.predict(input_df)[0]


    return jsonify({
        "co2": float(prediction)
    })

    


if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0", port=5000)