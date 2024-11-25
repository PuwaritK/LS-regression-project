import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import random

DATA_POINTS = 300


def retrieve_data():
    # download dataset from https://www.kaggle.com/datasets/beridzeg45/apartment-prices
    kagglehub.dataset_download("beridzeg45/apartment-prices")


def prep_data() -> list[list[float]]:
    # import data from .csv file
    df = pd.read_csv(
        "C://Users//Nigga//.cache//kagglehub//datasets//beridzeg45//apartment-prices//versions//2//Apartment Prices.csv"
    )

    # set seed to always random the same point when running the file.
    # the data picked is still random.
    random.seed(69)

    # sample 300 data points from 7259 records
    record_numbers = random.sample(range(0, 7258), k=DATA_POINTS)
    record = df.loc[record_numbers, ["PRICE (GEL)", "Area", "Rooms"]]

    # assign the values
    y = record["PRICE (GEL)"].to_list()
    x1 = record["Area"].to_list()
    x2 = record["Rooms"].to_list()
    return [x1, x2, y]


def calculate_sum_of_squared_errors(y_i: list[float], y_hat: list[float]) -> float:

    sse = 0
    for i in range(len(y_hat)):
        sse += (y_i[i] - y_hat[i]) ** 2
    return sse


def calculate_r_square(y_i: list[float], y_hat: list[float]) -> float:
    y_bar: float = sum(y_i) / len(y_i)

    ss_tot = 0
    for i in range(len(y_i)):
        ss_tot += (y_i[i] - y_bar) ** 2

    ss_res = 0
    for i in range(len(y_i)):
        ss_res += (y_i[i] - y_hat[i]) ** 2

    return (ss_tot - ss_res) / ss_tot


def calculate_root_mean_square_error(y_i: list[float], y_hat: list[float]) -> float:
    n = len(y_i)

    mse = 0
    for i in range(n):
        mse += ((y_hat[i] - y_i[i]) ** 2) / n

    rmse = mse**0.5
    return rmse


def linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    sx = sum(x)
    sxx = sum(np.power(x, 2))
    sy = sum(y)
    sxy = sum(np.multiply(x, y))
    a0 = (sxx * sy - sxy * sx) / (n * sxx - sx**2)
    a1 = (n * sxy - sx * sy) / (n * sxx - sx**2)
    return a0, a1


def polynomial_regression_degree2(
    x: list[float], y: list[float]
) -> tuple[float, float, float]:
    n = len(x)
    sum_x = sum(x)
    sum_x2 = sum(np.power(x, 2))
    sum_y = sum(y)
    sum_xy = sum(np.multiply(x, y))
    sum_x3 = sum(np.power(x, 3))
    sum_x4 = sum(np.power(x, 4))
    sum_x2y = sum(np.multiply(y, np.power(x, 2)))
    A = np.array(
        [[n, sum_x, sum_x2], [sum_x, sum_x2, sum_x3], [sum_x2, sum_x3, sum_x4]]
    )
    b = np.array([[sum_y], [sum_xy], [sum_x2y]])

    polynomial_sol = np.linalg.solve(A, b)
    a0 = polynomial_sol[0][0]
    a1 = polynomial_sol[1][0]
    a2 = polynomial_sol[2][0]
    return a0, a1, a2


def exponential_curve_fitting(x: list[float], y: list[float]) -> tuple[float, float]:
    big_y = np.log(y)
    big_x = x
    n = len(big_x)
    sx = sum(big_x)
    sxx = sum(np.power(big_x, 2))
    sy = sum(big_y)
    sxy = sum(np.multiply(big_x, big_y))
    a0 = (sxx * sy - sxy * sx) / (n * sxx - sx**2)
    a1 = (n * sxy - sx * sy) / (n * sxx - sx**2)

    b = np.exp(a0)
    m = a1

    return b, m


def main():
    x1, x2, _y = prep_data()
    # x1 is Area of the apartment, x2 is number of rooms in the apartment,
    # y is the price of the apartment

    # sorts x1 such that x1 is ascending
    x1, y1 = zip(*sorted(zip(x1, _y), key=lambda x: x[0]))

    # sorts x2 such that x2 is ascending
    x2, y2 = zip(*sorted(zip(x2, _y), key=lambda x: x[0]))

    # Linear Regression
    lr1_a0, lr1_a1 = linear_regression(x1, y1)
    lr2_a0, lr2_a1 = linear_regression(x2, y2)
    figLR, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 9))
    figLR.suptitle(f"Linear Regression Model ({DATA_POINTS} data points)")

    # calculate y_hat (predicted price)
    y_hat1 = []
    for i in range(DATA_POINTS):
        y_hat1.append(lr1_a0 + x1[i] * lr1_a1)

    y_hat2 = []
    for i in range(DATA_POINTS):
        y_hat2.append(lr2_a0 + x2[i] * lr2_a1)

    # configure plot
    ax1.scatter(x1, y1, s=60, alpha=0.7, edgecolors="k")
    ax1.plot(x1, y_hat1, color="k")
    ax1.ticklabel_format(style="plain")
    ax1.set_title("Area VS. Price")
    ax1.set_xlabel("Area (Square Meters)")
    ax1.set_ylabel("Price (Georgian Lari)")

    ax1_rsquare = calculate_r_square(y1, y_hat1)
    ax1.text(150, 50, f"R^2 = {ax1_rsquare}")

    ax2.scatter(x2, y2, s=60, alpha=0.7, edgecolors="k")
    ax2.plot(x2, y_hat2, color="k")
    ax2.set_title("Number of rooms VS. Price")
    ax2.set_xlabel("Number of rooms")
    ax2.set_ylabel("Price (Georgian Lari)")
    ax2.ticklabel_format(style="plain")

    ax2_rsquare = calculate_r_square(y2, y_hat2)
    ax2.text(4, 50, f"R^2 = {ax2_rsquare}")

    # sse
    print(
        f"Sum of squared errors of linear regression model's x1 is {calculate_sum_of_squared_errors(y1,y_hat1)}"
    )
    print(
        f"Sum of squared errors of linear regression model's x2 is {calculate_sum_of_squared_errors(y2,y_hat2)}"
    )

    # rmse
    print(
        f"Root mean square error of linear regression model's x1 is {calculate_root_mean_square_error(y1, y_hat1)}"
    )
    print(
        f"Root mean square error of linear regression model's x2 is {calculate_root_mean_square_error(y2, y_hat2)}\n"
    )
    # -------------------------------------------------------------------------------

    # Polynomial Regression Degree 2
    pr1_a0, pr1_a1, pr1_a2 = polynomial_regression_degree2(x1, y1)
    pr2_a0, pr2_a1, pr2_a2 = polynomial_regression_degree2(x2, y2)
    figPR, (ax3, ax4) = plt.subplots(1, 2, figsize=(9, 9))
    figPR.suptitle(f"Polynomial Regression Degree 2 Model ({DATA_POINTS} data points)")

    # calculate y_hat (predicted price)
    PR_y_hat1 = []
    for i in range(DATA_POINTS):
        PR_y_hat1.append(pr1_a0 + x1[i] * pr1_a1 + pr1_a2 * x1[i] ** 2)

    PR_y_hat2 = []
    for i in range(DATA_POINTS):
        PR_y_hat2.append(pr2_a0 + x2[i] * pr2_a1 + pr2_a2 * x2[i] ** 2)

    # configure plot
    ax3.scatter(x1, y1, s=60, alpha=0.7, edgecolors="k")
    ax3.plot(x1, PR_y_hat1, color="k")
    ax3.ticklabel_format(style="plain")
    ax3.set_title("Area VS. Price")
    ax3.set_xlabel("Area (Square Meters)")
    ax3.set_ylabel("Price (Georgian Lari)")

    ax3_rsquare = calculate_r_square(y1, PR_y_hat1)
    ax3.text(150, 50, f"R^2 = {ax3_rsquare}")

    ax4.scatter(x2, y2, s=60, alpha=0.7, edgecolors="k")
    ax4.plot(x2, PR_y_hat2, color="k")
    ax4.set_title("Number of rooms VS. Price")
    ax4.set_xlabel("Number of rooms")
    ax4.set_ylabel("Price (Georgian Lari)")
    ax4.ticklabel_format(style="plain")

    ax4_rsquare = calculate_r_square(y2, PR_y_hat2)
    ax4.text(4, 50, f"R^2 = {ax4_rsquare}")

    # sse
    print(
        f"Sum of squared errors of polynomial regression degree 2 model's x1 is {calculate_sum_of_squared_errors(y1,PR_y_hat1)}"
    )
    print(
        f"Sum of squared errors of polynomial regression degree 2 model's x2 is {calculate_sum_of_squared_errors(y2,PR_y_hat2)}"
    )

    # rmse
    print(
        f"Root mean square error of polynomial regression degree 2 model's x1 is {calculate_root_mean_square_error(y1, PR_y_hat1)}"
    )
    print(
        f"Root mean square error of polynomial regression degree 2 model's x2 is {calculate_root_mean_square_error(y2, PR_y_hat2)}\n"
    )
    # -------------------------------------------------------------------------------

    # Exponential
    b1, m1 = exponential_curve_fitting(x1, y1)
    b2, m2 = exponential_curve_fitting(x2, y2)
    figECF, (ax5, ax6) = plt.subplots(1, 2, figsize=(9, 9))
    figECF.suptitle(f"Exponential Function Model ({DATA_POINTS} data points)")

    # calculate y_hat (predicted price)
    ECF_y_hat1 = []
    for i in range(DATA_POINTS):
        ECF_y_hat1.append(b1 * np.e ** (m1 * x1[i]))

    ECF_y_hat2 = []
    for i in range(DATA_POINTS):
        ECF_y_hat2.append(b2 * np.e ** (m2 * x2[i]))

    # configure plot
    ax5.scatter(x1, y1, s=60, alpha=0.7, edgecolors="k")
    ax5.plot(x1, ECF_y_hat1, color="k")
    ax5.ticklabel_format(style="plain")
    ax5.set_title("Area VS. Price")
    ax5.set_xlabel("Area (Square Meters)")
    ax5.set_ylabel("Price (Georgian Lari)")

    ax5_rsquare = calculate_r_square(y1, ECF_y_hat1)
    ax5.text(150, 50, f"R^2 = {ax5_rsquare}")

    ax6.scatter(x2, y2, s=60, alpha=0.7, edgecolors="k")
    ax6.plot(x2, ECF_y_hat2, color="k")
    ax6.set_title("Number of rooms VS. Price")
    ax6.set_xlabel("Number of rooms")
    ax6.set_ylabel("Price (Georgian Lari)")
    ax6.ticklabel_format(style="plain")

    ax6_rsquare = calculate_r_square(y2, ECF_y_hat2)
    ax6.text(4, 50, f"R^2 = {ax6_rsquare}")

    # Provide some examples that use the constructed regression models for prediction.
    figEXP, ax7 = plt.subplots(figsize=(9, 9))

    # example of predictions of price when room area is 500, 750, 1000, 2500, 4500 using linear regression model
    xEXP = [250, 500, 750, 1000, 2000]

    # calculate y_hat_EXP
    y_hat_EXP = []
    for i in range(len(xEXP)):
        y_hat_EXP.append(lr1_a0 + xEXP[i] * lr1_a1)

    ax7.scatter(
        x1,
        y1,
        s=60,
        alpha=0.7,
        edgecolors="red",
    )
    ax7.scatter(xEXP, y_hat_EXP, s=60, alpha=0.7, edgecolors="blue")
    ax7.plot(x1, y_hat1, color="k")
    ax7.ticklabel_format(style="plain")
    ax7.set_title(
        "Area VS. Price (Dataset plotted with red outline, example set plotted with blue outline, \nusing the same regression line as dataset's)"
    )
    ax7.set_xlabel("Area (Square Meters)")
    ax7.set_ylabel("Price (Georgian Lari)")

    # sse
    print(
        f"Sum of squared errors of exponential regression model's x1 is {calculate_sum_of_squared_errors(y1,ECF_y_hat1)}"
    )
    print(
        f"Sum of squared errors of exponential regression model's x2 is {calculate_sum_of_squared_errors(y2,ECF_y_hat2)}"
    )

    # rmse
    print(
        f"Root mean square error of exponential regression model's x1 is {calculate_root_mean_square_error(y1, ECF_y_hat1)}"
    )
    print(
        f"Root mean square error of exponential regression model's x2 is {calculate_root_mean_square_error(y2, ECF_y_hat2)}\n"
    )

    # display linear regression figure
    plt.show()


if __name__ == "__main__":
    main()
