import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Spark Session
# ----------------------------

spark = SparkSession.builder \
    .appName("AI Dashboard") \
    .master("local[*]") \
    .getOrCreate()

# ----------------------------
# PyTorch Model
# ----------------------------

class SalesModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(1,1)

    def forward(self,x):

        return self.linear(x)


# ----------------------------
# Main App
# ----------------------------

class Dashboard:

    def __init__(self,root):

        self.root=root

        self.root.title("PySpark + PyTorch Dashboard")

        self.root.geometry("700x500")

        self.df=None

        tk.Button(root,text="Upload CSV",
                  command=self.upload,width=20,height=2).pack(pady=10)

        tk.Button(root,text="Train Model",
                  command=self.train,width=20,height=2).pack(pady=10)

        tk.Button(root,text="Show Graph",
                  command=self.graph,width=20,height=2).pack(pady=10)

        tk.Button(root,text="Export Prediction",
                  command=self.export,width=20,height=2).pack(pady=10)

        self.label=tk.Label(root,text="No Data",font=("Arial",12))

        self.label.pack(pady=20)

    # ------------------------

    def upload(self):

        file=filedialog.askopenfilename(filetypes=[("CSV","*.csv")])

        if file=="":

            return

        spark_df=spark.read.csv(file,header=True,inferSchema=True)

        spark_df=spark_df.dropna()

        spark_df=spark_df.filter(col("Sales")>0)

        self.df=spark_df.toPandas()

        self.label.config(text=f"{len(self.df)} Records Loaded")

    # ------------------------

    def train(self):

        if self.df is None:

            messagebox.showerror("Error","Upload CSV")

            return

        x=torch.tensor(self.df[['Day']].values,dtype=torch.float32)

        y=torch.tensor(self.df[['Sales']].values,dtype=torch.float32)

        model=SalesModel()

        criterion=nn.MSELoss()

        optimizer=optim.Adam(model.parameters(),lr=0.01)

        for epoch in range(1000):

            pred=model(x)

            loss=criterion(pred,y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        self.model=model

        prediction=model(x).detach().numpy()

        self.df["Prediction"]=prediction

        mse=((self.df["Sales"]-self.df["Prediction"])**2).mean()

        self.label.config(text=f"Training Complete\nMSE={mse:.2f}")

    # ------------------------

    def graph(self):

        if self.df is None:

            return

        plt.figure(figsize=(8,5))

        plt.plot(self.df["Day"],self.df["Sales"],label="Actual")

        plt.plot(self.df["Day"],self.df["Prediction"],label="Prediction")

        plt.legend()

        plt.title("Sales Prediction")

        plt.show()

    # ------------------------

    def export(self):

        if self.df is None:

            return

        file=filedialog.asksaveasfilename(defaultextension=".xlsx")

        self.df.to_excel(file,index=False)

        messagebox.showinfo("Done","Exported Successfully")


# ----------------------------

root=tk.Tk()

Dashboard(root)

root.mainloop()
