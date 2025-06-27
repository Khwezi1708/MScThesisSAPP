# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_val, y_val_pred, y_test, y_test_pred, X_val, X_test, plot_fig=None):
    # Helper function to calculate DAE (Daily Average Error)
    def calculate_dae(actual, predicted, df):
        df['actual'] = actual
        df['predicted'] = predicted

        # Ensure the index is a DatetimeIndex and use df.index.date for grouping
        df.index = pd.to_datetime(df.index)  # Convert index to DatetimeIndex if it's not already

        # Group by the date part of the index
        daily_actual = df.groupby(df.index.date)['actual'].mean()
        daily_pred = df.groupby(df.index.date)['predicted'].mean()
        daily_error = abs(daily_pred - daily_actual)
        return daily_error.mean()

    # Initialize a list to store the results
    results = []

    # Evaluate performance on the validation set using MAE
    mae_val = mean_absolute_error(y_val, y_val_pred)

    # Calculate Root Mean Squared Error (RMSE) on the validation set
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    # Ensure that the Date index is correctly added to val_df
    val_df = X_val.copy()  # Use X_val as the starting point
    val_df['actual'] = y_val
    val_df['predicted'] = y_val_pred

    # Calculate Daily Average Error (DAE) on the validation set
    dae_val = calculate_dae(y_val, y_val_pred, val_df)

    # Append the validation results to the list
    results.append({
        'Metric': 'Validation Set',
        'MAE': mae_val,
        'RMSE': rmse_val,
        'DAE': dae_val
    })

    # Evaluate performance on the test set using MAE
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate Root Mean Squared Error (RMSE) on the test set
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Ensure that the Date index is correctly added to test_df
    test_df = X_test.copy()  # Use X_test as the starting point
    test_df['actual'] = y_test
    test_df['predicted'] = y_test_pred

    # Calculate Daily Average Error (DAE) on the test set
    dae_test = calculate_dae(y_test, y_test_pred, test_df)

    # Append the test results to the list
    results.append({
        'Metric': 'Test Set',
        'MAE': mae_test,
        'RMSE': rmse_test,
        'DAE': dae_test
    })

    # Convert the results list to a DataFrame and return it
    results_df = pd.DataFrame(results)

    # If a plotting function is provided, execute it
    if plot_fig:
        plot_fig(y_test, y_test_pred)  # Execute the plot function
    return results_df

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_lstm(y_val, y_val_pred, y_test, y_test_pred, X_val, X_test, plot_fig=None):
    # Helper function to calculate DAE (Daily Average Error)
    def calculate_dae(actual, predicted, df):
        df['actual'] = actual
        df['predicted'] = predicted

        # Ensure the index is a DatetimeIndex and use df.index.date for grouping
        df.index = pd.to_datetime(df.index)  # Convert index to DatetimeIndex if it's not already

        # Group by the date part of the index
        daily_actual = df.groupby(df.index.date)['actual'].mean()
        daily_pred = df.groupby(df.index.date)['predicted'].mean()
        daily_error = abs(daily_pred - daily_actual)
        return daily_error.mean()

    # Function to calculate how often predictions are lower than actual prices
    def count_lower_predictions(actual, predicted):
        return np.sum(predicted < actual) / len(actual) * 100  # returns percentage of times predicted < actual

    # Initialize a list to store the results
    results = []

    # Evaluate performance on the validation set using MAE
    mae_val = mean_absolute_error(y_val, y_val_pred)

    # Calculate Root Mean Squared Error (RMSE) on the validation set
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    # Ensure that the Date index is correctly added to val_df
    val_df = X_val.copy()  # Use X_val as the starting point
    val_df['actual'] = y_val
    val_df['predicted'] = y_val_pred

    # Calculate Daily Average Error (DAE) on the validation set
    dae_val = calculate_dae(y_val, y_val_pred, val_df)

    # Explained variance score (R2) on the validation set
    r2_val = r2_score(y_val, y_val_pred)

    # Calculate how often predicted prices are lower than actual prices for the validation set
    lower_predictions_val = count_lower_predictions(y_val, y_val_pred)

    # Append the validation results to the list
    results.append({
        'Metric': 'Validation Set',
        'MAE': mae_val,
        'DAE': dae_val,
        'RMSE': rmse_val,
        'R2': r2_val,
        'LP (%)': lower_predictions_val
    })

    # Evaluate performance on the test set using MAE
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate Root Mean Squared Error (RMSE) on the test set
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Ensure that the Date index is correctly added to test_df
    test_df = X_test.copy()  # Use X_test as the starting point
    test_df['actual'] = y_test
    test_df['predicted'] = y_test_pred

    # Calculate Daily Average Error (DAE) on the test set
    dae_test = calculate_dae(y_test, y_test_pred, test_df)

    # Explained variance score (R2) on the test set
    r2_test = r2_score(y_test, y_test_pred)

    # Calculate how often predicted prices are lower than actual prices for the test set
    lower_predictions_test = count_lower_predictions(y_test, y_test_pred)

    # Append the test results to the list
    results.append({
        'Metric': 'Test Set',
        'MAE': mae_test,
        'RMSE': rmse_test,
        'DAE': dae_test,
        'R2': r2_test,
        'LP (%)': lower_predictions_test
    })


    # Convert the results list to a DataFrame and return it
    results_df = pd.DataFrame(results)
    # Print LaTeX rows with ampersands
    for idx, row in results_df.iterrows():
        
        print(f" & {row['Metric']} & {row['MAE']:.1f} & {row['DAE']:.1f} & {'x'} & {row['RMSE']:.1f} & {row['R2']:.2f} & {row['LP (%)']:.1f} \\\\")

    if plot_fig:
        plot_fig(y_test, y_test_pred)
    # If a plotting function is provided, execute it
    if plot_fig:
        plot_fig(y_test, y_test_pred)  # Execute the plot function

    return results_df


def plot_comparison(y_test, y_test_pred, df, plot_type='year'):
    # Ensure the 'Date' column is used from the DataFrame for correct time representation
    actual_dates = y_test.index  # Use the index of y_test as the time axis for plotting

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot Actual Prices
    plt.plot(actual_dates, y_test, label='Actual Prices', color='blue')

    # Plot Predicted Prices
    plt.plot(actual_dates, y_test_pred, label='Predicted Prices', color='red')

    # Add labels, title, and legend
    plt.xlabel('Date')
    plt.ylabel('Price (USD/MWh)')
    plt.title('Actual vs Predicted Prices')
    plt.legend()

    # Format the x-axis to show only year or month/year
    if plot_type == 'year':
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display only year
    elif plot_type == 'month':
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to each month
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Display month and year

    # Rotate date labels for readability
    plt.xticks(rotation=45)

    # Display the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


