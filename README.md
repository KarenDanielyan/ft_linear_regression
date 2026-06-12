# ft_linear_regression

A simple linear regression project to predict the price of a car based on its mileage.

## Usage

First, train the model using the provided dataset:

```bash
python train.py
```

Then, to predict a car's price, run the prediction script:

```bash
python predict.py
```

To see the model's accuracy and visualize the regression line, run the bonus script:

```bash
python bonus.py
```

## Files

* `train.py`: Reads the training dataset (`data.csv`), normalizes the mileage data, and uses gradient descent to calculate the optimal linear regression parameters. It then saves these parameters into `model.json` to be used for future predictions.
* `predict.py`: The main script that takes mileage as input and outputs the estimated price based on the saved model parameters.
* `bonus.py`: Evaluates the model's accuracy by calculating the R-squared (R²) score. It also generates a visual plot showing the original data points and the regression line calculated by the model.
* `model.json`: Stores the model's parameters:
  * `theta0` and `theta1`: The weights for the linear regression hypothesis.
  * `norm_min` and `norm_max`: The min/max normalization parameters used to scale the input mileage before predicting the price.

If `model.json` is missing, the script will safely default `theta0` and `theta1` to `0`.
