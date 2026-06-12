from flask import Flask, render_template, request # type: ignore
import pandas as pd
import joblib


app = Flask(__name__)
model = joblib.load("iris_rf_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])

        # Feature Engineering
        petal_ratio = pl / pw
        sepal_ratio = sl / sw

        total_length = sl + pl
        total_width = sw + pw


        data = pd.DataFrame([{
            "sepal length (cm)": sl,
            "sepal width (cm)": sw,
            "petal length (cm)": pl,
            "petal width (cm)": pw,
            "petal_ratio": petal_ratio,
            "sepal_ratio": sepal_ratio,
            "total_length": total_length,
            "total_width": total_width,
        }])

        pred = model.predict(data)[0]

        prediction = str(pred).title()  

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)