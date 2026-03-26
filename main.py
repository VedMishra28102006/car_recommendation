from flask import Flask, jsonify, render_template, request
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

def rs_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as fd:
            model = pickle.load(fd)
    else:
        df = pd.read_csv("train.csv")
        df = df.drop("ID", axis=1)
        df["Mileage"] = df["Mileage"].str.replace(" km", "", case=False).astype(float)
        df["Levy"] = df["Levy"].replace("-", 0).astype(float)
        df["Engine volume"] = df["Engine volume"].str.replace(" Turbo", "", case=False).astype(float)
        encoder = OneHotEncoder()
        scaler = StandardScaler()
        X_cat = encoder.fit_transform(df[[
            "Manufacturer", "Model", "Category", "Leather interior",
            "Fuel type", "Gear box type", "Drive wheels", "Doors",
            "Wheel", "Color"
        ]])
        X_num = scaler.fit_transform(df[[
            "Airbags", "Cylinders", "Engine volume", "Levy",
            "Mileage", "Price", "Prod. year"
        ]].values)
        X = np.hstack([X_cat.toarray(), X_num])
        model = {"df": df, "encoder": encoder, "scaler": scaler, "X": X}
        with open("model.pkl", "wb") as fd:
            pickle.dump(model, fd)
    return model

model = rs_model()
df = model["df"]
encoder = model["encoder"]
scaler = model["scaler"]
X = model["X"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/options", methods=["GET"])
def options():
    options = {}
    groups = df.groupby(["Manufacturer", "Model", "Category"])
    for (manufacturer, model_name, category), group in groups:
        colors = group["Color"].unique().tolist()
        if manufacturer not in options:
            options[manufacturer] = {}
        if model_name not in options[manufacturer]:
            options[manufacturer][model_name] = {}
        options[manufacturer][model_name][category] = colors
    return jsonify(options)

@app.route("/recommend", methods=["POST"])
def recommend():
    if request.form and set(["Manufacturer", "Model", "Category", "Color", "page"]).issubset(request.form.keys()):
        manufacturers = request.form.getlist("Manufacturer")
        models = request.form.getlist("Model")
        categories = request.form.getlist("Category")
        colors = request.form.getlist("Color")
        page = int(request.form.get("page"))
        match = df[
            df["Manufacturer"].isin(manufacturers)
            & df["Model"].isin(models)
            & df["Category"].isin(categories)
            & df["Color"].isin(colors)
        ]
        input_cat = encoder.transform(match[[
            "Manufacturer", "Model", "Category", "Leather interior",
            "Fuel type", "Gear box type", "Drive wheels", "Doors",
            "Wheel", "Color"
        ]])
        input_num = scaler.transform(match[[
            "Airbags", "Cylinders", "Engine volume", "Levy",
            "Mileage", "Price", "Prod. year"
        ]].values)
        input_vectors = np.hstack([input_cat.toarray(), input_num])
        input_vec = input_vectors.mean(axis=0).reshape(1, -1)
        similarities = cosine_similarity(input_vec, X).flatten()
        df["similarity"] = similarities
        cars = df[
            ~(
                df["Manufacturer"].isin(manufacturers)
                & df["Model"].isin(models)
                & df["Category"].isin(categories)
                & df["Color"].isin(colors)
            ) & (df["similarity"] > 0.5)
        ].sort_values(by="similarity", ascending=False).head(100)
        cars = cars.drop_duplicates(subset=["Manufacturer", "Model", "Category", "Color"])
        recommendations = cars.iloc[(page - 1) * 10:page * 10]
        has_more = 0 if cars.iloc[page * 10:(page + 1) * 10].empty else 1
        return jsonify({
            "Manufacturer": recommendations["Manufacturer"].tolist(),
            "Model": recommendations["Model"].tolist(),
            "Category": recommendations["Category"].tolist(),
            "Color": recommendations["Color"].tolist(),
            "has_more": has_more
        })
    return "", 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
