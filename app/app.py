import sys
import os

# let imports find the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, render_template, request

from src.predict import load_model, predict_one

app = Flask(__name__)

model = load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_form():
    features = {
        "age": request.form["age"],
        "tenure": request.form["tenure"],
        "monthly_charges": request.form["monthly_charges"],
        "total_charges": request.form["total_charges"],
        "gender": request.form["gender"],
        "contract": request.form["contract"],
        "payment_method": request.form["payment_method"],
        "internet_service": request.form["internet_service"],
        "online_security": request.form["online_security"],
        "tech_support": request.form["tech_support"],
    }

    result = predict_one(model, features)
    return render_template("index.html", result=result, data=features)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    features = request.get_json()
    result = predict_one(model, features)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
