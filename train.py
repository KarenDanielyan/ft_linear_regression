# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kdaniely <kdaniely@student.42yerevan.am    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/06/12 14:16:09 by kdaniely          #+#    #+#              #
#    Updated: 2026/06/12 14:16:09 by kdaniely         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Imports
import numpy as np
import pandas as pd


# helper functions
def normalize(entry, min_val, max_val):
    return (entry - min_val) / (max_val - min_val)


# main
if __name__ == "__main__":
    # Initialize variables
    data = pd.read_csv("data.csv")
    price = data["price"]
    mileage = data["km"]
    min_val = mileage.min()
    max_val = mileage.max()
    mileage_norm = mileage.apply(lambda x: normalize(x, min_val, max_val))
    # Learning params
    learning_rate = 0.1
    epochs = 1500
    m = len(data)
    theta_0 = 0
    theta_1 = 0
    # Training loop
    for epoch in range(epochs):
        sum_error_0 = 0
        sum_error_1 = 0
        for i in range(m):
            # 1. Calculate the prediction using scaled mileage
            prediction = theta_0 + (theta_1 * mileage_norm[i])
            # 2. Calculate the raw error
            error = prediction - price[i]
            # 3. Accumulate the errors for the gradients
            sum_error_0 += error
            sum_error_1 += error * mileage_norm[i]
            # 4. Compute the temporary gradient steps
            tmp_theta_0 = learning_rate * (1 / m) * sum_error_0
            tmp_theta_1 = learning_rate * (1 / m) * sum_error_1
            # 5. Simultaneously update the weights using gradient DESCENT
            theta_0 = theta_0 - tmp_theta_0
            theta_1 = theta_1 - tmp_theta_1
    print("Training complete")
    print(f"theta_0: {theta_0}, theta_1: {theta_1}")
    # Save theta in a file
