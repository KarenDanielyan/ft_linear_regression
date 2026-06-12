# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    predict.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kdaniely <kdaniely@student.42yerevan.am    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/06/12 14:16:37 by kdaniely          #+#    #+#              #
#    Updated: 2026/06/12 14:16:37 by kdaniely         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import json
import os
import sys


def predict_price():
    theta0 = 0.0
    theta1 = 0.0
    norm_min = 0.0
    norm_max = 1.0

    model_file = "model.json"

    if not os.path.exists(model_file):
        print("Warning: model.json not found. Assuming theta0 = 0 and theta1 = 0.")
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
        except Exception as e:
            print(f"Warning: Could not read {model_file} correctly. Error: {e}")
            print("Assuming theta0 = 0 and theta1 = 0.")

    try:
        mileage_str = input("Please enter a mileage: ")
        mileage = float(mileage_str)
    except ValueError:
        print("Error: Invalid mileage. Please enter a valid number.")
        sys.exit(1)
    except EOFError:
        print("\nExiting...")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

    # Normalize the input mileage
    if norm_max - norm_min != 0:
        normalized_mileage = (mileage - norm_min) / (norm_max - norm_min)
    else:
        normalized_mileage = 0.0

    # Predict price
    estimated_price = theta0 + (theta1 * normalized_mileage)

    # We only format to round to reasonable decimal,
    # or just let python print float
    print(f"The estimated price for {mileage} mileage is: {estimated_price}")


if __name__ == "__main__":
    predict_price()
