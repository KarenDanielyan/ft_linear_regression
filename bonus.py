# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    bonus.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kdaniely <kdaniely@student.42yerevan.am    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/06/12 14:16:56 by kdaniely          #+#    #+#              #
#    Updated: 2026/06/12 14:16:56 by kdaniely         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Imports
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from predict import predict_price


def calculate_r_squared(mileage, price, theta_0, theta_1, norm_min, norm_max):
    m = len(price)

    # 1. Calculate the mean of the actual prices (baseline model prediction)
    mean_price = sum(price) / m

    ss_res = 0
    ss_tot = 0
    for i in range(m):
        # Model prediction using your optimized weights
        prediction = predict_price(mileage[i], theta_0, theta_1, norm_min, norm_max)
        # Accumulate squared errors
        ss_res += (price[i] - prediction) ** 2
        ss_tot += (price[i] - mean_price) ** 2

    # Prevent division by zero if all car prices are identical
    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    model_file = "model.json"
    df = pd.read_csv("data.csv")
    mileage = df["km"].values
    price = df["price"].values

    if not os.path.exists(model_file):
        print("Warning: model.json not found. Assuming theta0 = 0 and theta1 = 0.")
        theta0 = 0.0
        theta1 = 0.0
        norm_min = 0.0
        norm_max = 1.0
    else:
        try:
            with open(model_file, "r") as f:
                model_data = json.load(f)
                theta0 = float(model_data.get("theta_0", 0.0))
                theta1 = float(model_data.get("theta_1", 0.0))
                # Only use min/max if they exist in the model file
                if "norm_min" in model_data and "norm_max" in model_data:
                    norm_min = float(model_data["norm_min"])
                    norm_max = float(model_data["norm_max"])
                else:
                    norm_min = 0.0
                    norm_max = 1.0
        except Exception as e:
            print(f"Warning: Could not read {model_file} correctly. Error: {e}")
            print("Assuming theta0 = 0 and theta1 = 0.")
            theta0 = 0.0
            theta1 = 0.0
            norm_min = 0.0
            norm_max = 1.0

    r2 = calculate_r_squared(mileage, price, theta0, theta1, norm_min, norm_max)
    print(f"Model Precision (R² Score): {r2:.4f}")
    # Plot the data and the regression line
    plt.scatter(df["km"], df["price"], color="blue", label="Data")
    plt.plot(
        df["km"],
        predict_price(df["km"], theta0, theta1, norm_min, norm_max),
        color="red",
        label="Regression Line",
    )
    plt.text(0.8, 0.8, f"R² = {r2:.4f}", transform=plt.gca().transAxes)
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car Price Prediction")
    plt.legend()
    plt.show()
