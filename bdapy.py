
import pandas as pd
import numpy as np

from flask import Flask, render_template, request

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, when, sum as spark_sum

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



app = Flask(__name__)



spark = SparkSession.builder.appName("DemandPrediction").getOrCreate()



df = spark.read.csv("online_retail_II.csv", header=True, inferSchema=True)



df = df.dropna(subset=["Description","Country"])
df = df.filter(col("Quantity") > 0)



df = df.withColumn("InvoiceDate", col("InvoiceDate").cast("timestamp"))

df = df.withColumn("Month", month("InvoiceDate"))

df = df.withColumn(
    "Season",
    when(col("Month").isin([3,4,5,6]),"Summer")
    .when(col("Month").isin([7,8,9]),"Monsoon")
    .otherwise("Winter")
)



demand_df = df.groupBy(
    "Country","Season","Description"
).agg(
    spark_sum("Quantity").alias("Quantity")
)

demand_df = demand_df.toPandas()



le_country = LabelEncoder()
le_season = LabelEncoder()
le_product = LabelEncoder()

demand_df["Country_enc"] = le_country.fit_transform(demand_df["Country"])
demand_df["Season_enc"] = le_season.fit_transform(demand_df["Season"])
demand_df["Product_enc"] = le_product.fit_transform(demand_df["Description"])



X = demand_df[["Country_enc","Season_enc","Product_enc"]]
y = demand_df["Quantity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)



def recommend_products(country,season,top_n=10):

    country = country.strip()

    if country not in le_country.classes_:

        return pd.DataFrame({
            "Product":["Invalid Country"],
            "Predicted_Demand":["Country not found in dataset"]
        })

    country_enc = le_country.transform([country])[0]
    season_enc = le_season.transform([season])[0]

    products = demand_df["Product_enc"].unique()

    prediction_data = pd.DataFrame({
        "Country_enc": country_enc,
        "Season_enc": season_enc,
        "Product_enc": products
    })

    prediction_data["Predicted_Demand"] = model.predict(prediction_data)

    prediction_data["Product"] = le_product.inverse_transform(
        prediction_data["Product_enc"]
    )

    top_products = prediction_data.sort_values(
        "Predicted_Demand",ascending=False
    ).head(top_n)

    return top_products



@app.route("/")
def home():

    countries = sorted(demand_df["Country"].unique())

    return render_template(
        "bda.html",
        countries=countries
    )



@app.route("/predict", methods=["POST"])
def predict():

    country = request.form["country"]
    season = request.form["season"]

    results = recommend_products(country,season)

    output = results[["Product","Predicted_Demand"]].to_string(index=False)

    countries = sorted(demand_df["Country"].unique())

    return render_template(
        "bda.html",
        prediction_text=output,
        countries=countries
    )



if __name__ == "__main__":
    app.run(debug=True)