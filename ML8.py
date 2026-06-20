import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Create Spark Session
# -------------------------

spark = (
    SparkSession.builder
    .appName("AI Sales Forecast")
    .master("local[*]")
    .getOrCreate()
)

# -------------------------
# Main GUI Class
# -------------------------

class Dashboard:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Sales Forecast Dashboard")
        self.root.geometry("750x550")

        self.df = None
        self.model = None

        tk.Button(
            root,
            text="Upload CSV",
            width=25,
            command=self.upload
        ).pack(pady=10)

        tk.Button(
            root,
            text="Train Model",
            width=25,
            command=self.train
        ).pack(pady=10)

        tk.Button(
            root,
            text="Show Graph",
            width=25,
            command=self.graph
        ).pack(pady=10)

        tk.Button(
            root,
            text="Export Prediction",
            width=25,
            command=self.export
        ).pack(pady=10)

        self.status = tk.Label(
            root,
            text="Upload a CSV file",
            font=("Arial", 12)
        )

        self.status.pack(pady=20)

    # -------------------------

    def upload(self):

        file = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if file == "":
            return

        spark_df = spark.read.csv(
            file,
            header=True,
            inferSchema=True
        )

        spark_df = spark_df.dropna()

        spark_df = spark_df.filter(col("Sales") > 0)

        self.df = spark_df.toPandas()

        self.status.config(
            text=f"{len(self.df)} Records Loaded"
        )

    # -------------------------

    def train(self):

        if self.df is None:

            messagebox.showerror(
                "Error",
                "Please upload CSV first."
            )

            return

        X = self.df[["Day"]]

        y = self.df["Sales"]

        self.model = LinearRegression()

        self.model.fit(X, y)

        prediction = self.model.predict(X)

        self.df["Prediction"] = prediction

        mse = mean_squared_error(y, prediction)

        r2 = r2_score(y, prediction)

        self.status.config(
            text=f"Training Completed\nMSE = {mse:.2f}\nR² = {r2:.4f}"
        )

    # -------------------------

    def graph(self):

        if self.df is None:

            return

        plt.figure(figsize=(8,5))

        plt.plot(
            self.df["Day"],
            self.df["Sales"],
            marker="o",
            label="Actual"
        )

        plt.plot(
            self.df["Day"],
            self.df["Prediction"],
            marker="s",
            label="Predicted"
        )

        plt.title("Sales Forecast")

        plt.xlabel("Day")

        plt.ylabel("Sales")

        plt.legend()

        plt.grid(True)

        plt.show()

    # -------------------------

    def export(self):

        if self.df is None:

            return

        file = filedialog.asksaveasfilename(
            defaultextension=".xlsx"
        )

        if file == "":
            return

        self.df.to_excel(
            file,
            index=False
        )

        messagebox.showinfo(
            "Success",
            "Prediction exported successfully."
        )

# -------------------------

root = tk.Tk()

Dashboard(root)

root.mainloop()
