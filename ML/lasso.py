import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from config import features,path,start,end
import matplotlib.pyplot as plt



data = pd.read_csv(path)
values = data[features].values


# Construct time series supervised learning data:
def series_to_supervised(data, features, time_steps, step_ahead, target_col='software_quality_maintainability_remediation_effort'):
    """
    Parameters:
    data: pandas.DataFrame，containing all feature and target columns
    input_features: list of feature column names
    target_col: target column name
    time_steps: number of past weeks to use as input
    step_ahead: predict data for week t+step_ahead
    """
    # Use only input_features as model input
    input_features = [f for f in features if f != target_col]

    X, y = [], []
    for i in range(len(data) - time_steps - step_ahead + 1):
        x_seq = data[input_features].iloc[i: i + time_steps].values
        target = data[target_col].iloc[i + time_steps + step_ahead - 1]
        X.append(x_seq)
        y.append(target)
    return np.array(X), np.array(y)


steps = 1

alphas = [0.1, 0.01, 0.001]

best_results_df = pd.DataFrame(columns=['ahead', 'alpha', 'mae', 'rmse', 'mape'])
all_results_df = pd.DataFrame(columns=['ahead', 'alpha', 'mae', 'rmse', 'mape'])

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")


for ahead in range(start, end):
    print(f"\n=== Prediction period = {ahead} ===")

    X, y = series_to_supervised(data, features, time_steps=steps, step_ahead=ahead, target_col='sqale_index')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

    X_train_2d = X_train.reshape(-1, len(features) - 1)
    X_test_2d = X_test.reshape(-1, len(features) - 1)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_2d)
    X_test_scaled = scaler_X.transform(X_test_2d)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))


    results = []

 # Record the results of all parameter combinations
    for alpha in alphas:

        regressor = Lasso(alpha=alpha, random_state=0)
        regressor.fit(X_train_scaled, y_train_scaled.ravel())

        y_pred_scaled = regressor.predict(X_test_scaled)

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_test_orig = scaler_y.inverse_transform(y_test_scaled)

        mae = round(mean_absolute_error(y_test_orig, y_pred), 3)
        rmse = round(np.sqrt(mean_squared_error(y_test_orig, y_pred)), 3)
        mape = round((np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100), 3)

        results.append({
            'alpha': alpha,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'y_pred': y_pred.flatten(),
            'y_test': y_test_orig.flatten()
        })

        all_results_df = pd.concat([all_results_df, pd.DataFrame({
            'ahead': [ahead],
            'alpha': [alpha],
            'mae': [mae],
            'rmse': [rmse],
            'mape': [mape]
        })], ignore_index=True)

    best_result = min(results, key=lambda x: x['mape'])

    print(f"best alpha: {best_result['alpha']}")
    print(f"MAE: {best_result['mae']:.3f}")
    print(f"RMSE: {best_result['rmse']:.3f}")
    print(f"MAPE: {best_result['mape']:.3f}")

    best_results_df = pd.concat([best_results_df, pd.DataFrame({
        'ahead': [ahead],
        'alpha': [best_result['alpha']],
        'mae': [best_result['mae']],
        'rmse': [best_result['rmse']],
        'mape': [best_result['mape']]
    })], ignore_index=True)

    print("\n all results:")
    for result in results:
        print(
            f"Alpha={result['alpha']}: MAE={result['mae']:.3f}, RMSE={result['rmse']:.3f}, MAPE={result['mape']:.3f}")

# Plotting predicted values ​​and actual values. 
    # plt.figure(figsize=(12, 6))
    # plt.plot(best_result['y_test'], label='True Values', marker='o', linewidth=2)
    # plt.plot(best_result['y_pred'], label='Predicted Values', marker='s', linewidth=2)
    # plt.title(f'Lasso Regression: True vs Predicted Values (Ahead={ahead}, Alpha={best_result["alpha"]})')
    # plt.xlabel('Sample Index')
    # plt.ylabel('sqale_index')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # # save fig to desktop
    # plot_path = os.path.join(desktop_path, f"lasso_prediction_ahead_{ahead}.png")
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # print(f"The prediction results image has been saved to: {plot_path}")

# save results
best_results_path = os.path.join(desktop_path, "l1_best_results.csv")
all_results_path = os.path.join(desktop_path, "l1_all_results.csv")

best_results_df.to_csv(best_results_path, index=False)
all_results_df.to_csv(all_results_path, index=False)

print(f"\n Best result have been saved : {best_results_path}")
print(f"All results have been saved : {all_results_path}")