import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from config import features,path,start,end
import matplotlib.pyplot as plt


data = pd.read_csv(path)


def series_to_supervised(data, features, time_steps, step_ahead, target_col='sqale_index'):
    input_features = [f for f in features if f != target_col]
    X, y = [], []
    for i in range(len(data) - time_steps - step_ahead + 1):
        x_seq = data[input_features].iloc[i: i + time_steps].values
        target = data[target_col].iloc[i + time_steps + step_ahead - 1]
        X.append(x_seq)
        y.append(target)
    return np.array(X), np.array(y)


steps = 1


results_df = pd.DataFrame(columns=['ahead', 'mae', 'rmse', 'mape'])

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

for ahead in range(start, end):

    X, y = series_to_supervised(data, features, time_steps=steps, step_ahead=ahead, target_col='sqale_index')


    if len(X) == 0:
        print(f"Warning: No data generated for ahead={ahead}")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

    X_train_2d = X_train.reshape(-1, len(features) - 1)
    X_test_2d = X_test.reshape(-1, len(features) - 1)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_2d)
    X_test_scaled = scaler_X.transform(X_test_2d)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    regressor = LinearRegression()
    regressor.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = regressor.predict(X_test_scaled)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_actual = scaler_y.inverse_transform(y_test_scaled)

    mae = round(mean_absolute_error(y_test_actual, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test_actual, y_pred)), 3)
    mape = round((np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100), 3)

    results_df = pd.concat([results_df, pd.DataFrame({
        'ahead': [ahead],
        'mae': [mae],
        'rmse': [rmse],
        'mape': [mape]
    })], ignore_index=True)

    print(f"\n=== prediction period = {ahead} ===")
    print(f" MAE: {mae:.3f}")
    print(f" RMSE: {rmse:.3f}")
    print(f" MAPE: {mape:.3f}")

    # # Plotting predicted values ​​and actual values.
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test_actual.flatten(), label='True Values', marker='o', linewidth=2)
    # plt.plot(y_pred.flatten(), label='Predicted Values', marker='s', linewidth=2)
    # plt.title(f'Linear Regression: True vs Predicted Values (Ahead={ahead})')
    # plt.xlabel('Sample Index')
    # plt.ylabel('sqale_index')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # # save fig to desktop
    # plot_path = os.path.join(desktop_path, f"linear_regression_prediction_ahead_{ahead}.png")
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # print(f"The prediction results image has been saved to: {plot_path}")

# 保存结果到CSV文件
results_path = os.path.join(desktop_path, "linear_regression_results.csv")
results_df.to_csv(results_path, index=False)

print(f"\nAll results have been saved: {results_path}")