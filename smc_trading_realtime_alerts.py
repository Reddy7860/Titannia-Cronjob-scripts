import yfinance as yf
import pandas as pd
import numpy as numpy
from datetime import datetime

us_stocks_data = pd.read_csv("US_30_Stocks.csv")
main_pd_array = pd.read_csv('PD_Array_Data.csv')

for idx in range(len(us_stocks_data)):
    ticker = us_stocks_data.loc[idx, 'Symbol']
    # Download recent minute-level data to get minute-by-minute close prices
    minute_data = yf.download(ticker, interval="1m", period="1d")

    # Ensure the Datetime column exists in minute_data
    minute_data.reset_index(inplace=True)

    # Filter data for the current stock
    filter_pd_array = main_pd_array[main_pd_array['Stock'] == ticker]


    ## Checking for monthly

    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_FVG_Above_Top'].isna().any() and not filter_pd_array['monthly_FVG_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['monthly_FVG_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['monthly_FVG_Above_Bottom'].iloc[0]
        fvg_above_date = filter_pd_array['monthly_FVG_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: FVG Above The following High prices and their corresponding {fvg_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])

    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_FVG_Below_Top'].isna().any() and not filter_pd_array['monthly_FVG_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['monthly_FVG_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['monthly_FVG_Below_Bottom'].iloc[0]
        fvg_below_date = filter_pd_array['monthly_FVG_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: FVG Below The following Low prices and their corresponding {fvg_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_Swing_High'].isna().any():
        # Extract the top and bottom values
        swing_high = filter_pd_array['monthly_Swing_High'].iloc[0]
        swing_high_date = filter_pd_array['monthly_Swing_High_Date'].iloc[0]

        # Filter rows where the High price above the Swing High
        filtered_closes = minute_data[(minute_data['High'] >= swing_high)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: Swing High The following High prices and their corresponding {swing_high_date} are above {swing_high}:")
            print(f"Existing Range : {swing_high}")
            print(filtered_closes[['High', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_Swing_Low'].isna().any():
        # Extract the top and bottom values
        swing_low = filter_pd_array['monthly_Swing_Low'].iloc[0]
        swing_low_date = filter_pd_array['monthly_Swing_Low_Date'].iloc[0]

        # Filter rows where the Low price above the Swing Low
        filtered_closes = minute_data[(minute_data['Low'] <= swing_low)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: Swing Low The following Low prices and their corresponding {swing_low_date} are below {swing_low}:")
            print(f"Existing Range : {swing_low}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_OB_Above_Top'].isna().any() and not filter_pd_array['monthly_OB_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['monthly_OB_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['monthly_OB_Above_Bottom'].iloc[0]
        ob_above_date = filter_pd_array['monthly_OB_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: OB Above The following High prices and their corresponding {ob_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_OB_Below_Top'].isna().any() and not filter_pd_array['monthly_OB_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['monthly_OB_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['monthly_OB_Below_Bottom'].iloc[0]
        ob_below_date = filter_pd_array['monthly_OB_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: OB Below The following Low prices and their corresponding {ob_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_Breaker_OB_Above_Top'].isna().any() and not filter_pd_array['monthly_Breaker_OB_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['monthly_Breaker_OB_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['monthly_Breaker_OB_Above_Bottom'].iloc[0]
        ob_above_date = filter_pd_array['monthly_Breaker_OB_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: Breaker OB Above The following High prices and their corresponding {ob_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_Breaker_OB_Below_Top'].isna().any() and not filter_pd_array['monthly_Breaker_OB_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['monthly_Breaker_OB_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['monthly_Breaker_OB_Below_Bottom'].iloc[0]
        ob_below_date = filter_pd_array['monthly_Breaker_OB_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: Breaker OB Below The following Low prices and their corresponding {ob_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_Liquidity_untouched_Above'].isna().any():
        # Extract the top and bottom values
        liquidity_above = filter_pd_array['monthly_Liquidity_untouched_Above'].iloc[0]
        liquidity_above_date = filter_pd_array['monthly_Liquidity_untouched_Above_Date'].iloc[0]

        # Filter rows where the High price above the Swing High
        filtered_closes = minute_data[(minute_data['High'] >= liquidity_above)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: Liquidity Untouched The following High prices and their corresponding {liquidity_above_date} are above {liquidity_above}:")
            print(f"Existing Range : {liquidity_above}")
            print(filtered_closes[['High', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['monthly_Liquidity_untouched_Below'].isna().any():
        # Extract the top and bottom values
        liquidity_below = filter_pd_array['monthly_Liquidity_untouched_Below'].iloc[0]
        liquidity_liquidity_below_date = filter_pd_array['monthly_Liquidity_untouched_Below_Date'].iloc[0]

        # Filter rows where the Low price below the Liquidity
        filtered_closes = minute_data[(minute_data['Low'] <= liquidity_below)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"monthly: {ticker}: Liquidity Untouched The following Low prices and their corresponding {liquidity_liquidity_below_date} are below {liquidity_below}:")
            print(f"Existing Range : {liquidity_below}")
            print(filtered_closes[['Low', 'Datetime']])

    
    ## Checking for weekly
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_FVG_Above_Top'].isna().any() and not filter_pd_array['weekly_FVG_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['weekly_FVG_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['weekly_FVG_Above_Bottom'].iloc[0]
        fvg_above_date = filter_pd_array['weekly_FVG_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: FVG Above The following High prices and their corresponding {fvg_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])

    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_FVG_Below_Top'].isna().any() and not filter_pd_array['weekly_FVG_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['weekly_FVG_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['weekly_FVG_Below_Bottom'].iloc[0]
        fvg_below_date = filter_pd_array['weekly_FVG_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: FVG Below The following Low prices and their corresponding {fvg_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_Swing_High'].isna().any():
        # Extract the top and bottom values
        swing_high = filter_pd_array['weekly_Swing_High'].iloc[0]
        swing_high_date = filter_pd_array['weekly_Swing_High_Date'].iloc[0]

        # Filter rows where the High price above the Swing High
        filtered_closes = minute_data[(minute_data['High'] >= swing_high)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: Swing High The following High prices and their corresponding {swing_high_date} are above {swing_high}:")
            print(f"Existing Range : {swing_high}")
            print(filtered_closes[['High', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_Swing_Low'].isna().any():
        # Extract the top and bottom values
        swing_low = filter_pd_array['weekly_Swing_Low'].iloc[0]
        swing_low_date = filter_pd_array['weekly_Swing_Low_Date'].iloc[0]

        # Filter rows where the Low price above the Swing Low
        filtered_closes = minute_data[(minute_data['Low'] <= swing_low)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: Swing Low The following Low prices and their corresponding {swing_low_date} are below {swing_low}:")
            print(f"Existing Range : {swing_low}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_OB_Above_Top'].isna().any() and not filter_pd_array['weekly_OB_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['weekly_OB_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['weekly_OB_Above_Bottom'].iloc[0]
        ob_above_date = filter_pd_array['weekly_OB_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: OB Above The following High prices and their corresponding {ob_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_OB_Below_Top'].isna().any() and not filter_pd_array['weekly_OB_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['weekly_OB_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['weekly_OB_Below_Bottom'].iloc[0]
        ob_below_date = filter_pd_array['weekly_OB_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: OB Below The following Low prices and their corresponding {ob_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_Breaker_OB_Above_Top'].isna().any() and not filter_pd_array['weekly_Breaker_OB_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['weekly_Breaker_OB_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['weekly_Breaker_OB_Above_Bottom'].iloc[0]
        ob_above_date = filter_pd_array['weekly_Breaker_OB_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: Breaker OB Above The following High prices and their corresponding {ob_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_Breaker_OB_Below_Top'].isna().any() and not filter_pd_array['weekly_Breaker_OB_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['weekly_Breaker_OB_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['weekly_Breaker_OB_Below_Bottom'].iloc[0]
        ob_below_date = filter_pd_array['weekly_Breaker_OB_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: Breaker OB Below The following Low prices and their corresponding {ob_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_Liquidity_untouched_Above'].isna().any():
        # Extract the top and bottom values
        liquidity_above = filter_pd_array['weekly_Liquidity_untouched_Above'].iloc[0]
        liquidity_above_date = filter_pd_array['weekly_Liquidity_untouched_Above_Date'].iloc[0]

        # Filter rows where the High price above the Swing High
        filtered_closes = minute_data[(minute_data['High'] >= liquidity_above)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: Liquidity Untouched The following High prices and their corresponding {liquidity_above_date} are above {liquidity_above}:")
            print(f"Existing Range : {liquidity_above}")
            print(filtered_closes[['High', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['weekly_Liquidity_untouched_Below'].isna().any():
        # Extract the top and bottom values
        liquidity_below = filter_pd_array['weekly_Liquidity_untouched_Below'].iloc[0]
        liquidity_liquidity_below_date = filter_pd_array['weekly_Liquidity_untouched_Below_Date'].iloc[0]

        # Filter rows where the Low price below the Liquidity
        filtered_closes = minute_data[(minute_data['Low'] <= liquidity_below)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"weekly: {ticker}: Liquidity Untouched The following Low prices and their corresponding {liquidity_liquidity_below_date} are below {liquidity_below}:")
            print(f"Existing Range : {liquidity_below}")
            print(filtered_closes[['Low', 'Datetime']])

    ## Checking for daily
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_FVG_Above_Top'].isna().any() and not filter_pd_array['daily_FVG_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['daily_FVG_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['daily_FVG_Above_Bottom'].iloc[0]
        fvg_above_date = filter_pd_array['daily_FVG_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: FVG Above The following High prices and their corresponding {fvg_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])

    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_FVG_Below_Top'].isna().any() and not filter_pd_array['daily_FVG_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['daily_FVG_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['daily_FVG_Below_Bottom'].iloc[0]
        fvg_below_date = filter_pd_array['daily_FVG_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: FVG Below The following Low prices and their corresponding {fvg_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_Swing_High'].isna().any():
        # Extract the top and bottom values
        swing_high = filter_pd_array['daily_Swing_High'].iloc[0]
        swing_high_date = filter_pd_array['daily_Swing_High_Date'].iloc[0]

        # Filter rows where the High price above the Swing High
        filtered_closes = minute_data[(minute_data['High'] >= swing_high)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: Swing High The following High prices and their corresponding {swing_high_date} are above {swing_high}:")
            print(f"Existing Range : {swing_high}")
            print(filtered_closes[['High', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_Swing_Low'].isna().any():
        # Extract the top and bottom values
        swing_low = filter_pd_array['daily_Swing_Low'].iloc[0]
        swing_low_date = filter_pd_array['daily_Swing_Low_Date'].iloc[0]

        # Filter rows where the Low price above the Swing Low
        filtered_closes = minute_data[(minute_data['Low'] <= swing_low)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: Swing Low The following Low prices and their corresponding {swing_low_date} are below {swing_low}:")
            print(f"Existing Range : {swing_low}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_OB_Above_Top'].isna().any() and not filter_pd_array['daily_OB_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['daily_OB_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['daily_OB_Above_Bottom'].iloc[0]
        ob_above_date = filter_pd_array['daily_OB_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: OB Above The following High prices and their corresponding {ob_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_OB_Below_Top'].isna().any() and not filter_pd_array['daily_OB_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['daily_OB_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['daily_OB_Below_Bottom'].iloc[0]
        ob_below_date = filter_pd_array['daily_OB_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: OB Below The following Low prices and their corresponding {ob_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_Breaker_OB_Above_Top'].isna().any() and not filter_pd_array['daily_Breaker_OB_Above_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['daily_Breaker_OB_Above_Top'].iloc[0]
        bottom_value = filter_pd_array['daily_Breaker_OB_Above_Bottom'].iloc[0]
        ob_above_date = filter_pd_array['daily_Breaker_OB_Above_Date'].iloc[0]

        # Filter rows where the High price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['High'] >= bottom_value) & (minute_data['High'] <= top_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: Breaker OB Above The following High prices and their corresponding {ob_above_date} are between {bottom_value} and {top_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['High', 'Datetime']])
    
    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_Breaker_OB_Below_Top'].isna().any() and not filter_pd_array['daily_Breaker_OB_Below_Bottom'].isna().any():
        # Extract the top and bottom values
        top_value = filter_pd_array['daily_Breaker_OB_Below_Top'].iloc[0]
        bottom_value = filter_pd_array['daily_Breaker_OB_Below_Bottom'].iloc[0]
        ob_below_date = filter_pd_array['daily_Breaker_OB_Below_Date'].iloc[0]

        # Filter rows where the Low price is between the top and bottom values
        filtered_closes = minute_data[(minute_data['Low'] >= top_value) & (minute_data['Low'] <= bottom_value)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: Breaker OB Below The following Low prices and their corresponding {ob_below_date} are between {top_value} and {bottom_value}:")
            print(f"Existing Range : {top_value} and {bottom_value}")
            print(filtered_closes[['Low', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_Liquidity_untouched_Above'].isna().any():
        # Extract the top and bottom values
        liquidity_above = filter_pd_array['daily_Liquidity_untouched_Above'].iloc[0]
        liquidity_above_date = filter_pd_array['daily_Liquidity_untouched_Above_Date'].iloc[0]

        # Filter rows where the High price above the Swing High
        filtered_closes = minute_data[(minute_data['High'] >= liquidity_above)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: Liquidity Untouched The following High prices and their corresponding {liquidity_above_date} are above {liquidity_above}:")
            print(f"Existing Range : {liquidity_above}")
            print(filtered_closes[['High', 'Datetime']])

    # Check if both columns do not contain NaN values
    if not filter_pd_array['daily_Liquidity_untouched_Below'].isna().any():
        # Extract the top and bottom values
        liquidity_below = filter_pd_array['daily_Liquidity_untouched_Below'].iloc[0]
        liquidity_liquidity_below_date = filter_pd_array['daily_Liquidity_untouched_Below_Date'].iloc[0]

        # Filter rows where the Low price below the Liquidity
        filtered_closes = minute_data[(minute_data['Low'] <= liquidity_below)]

        # Print the datetime and close prices where the condition is satisfied
        if not filtered_closes.empty:
            print(f"daily: {ticker}: Liquidity Untouched The following Low prices and their corresponding {liquidity_liquidity_below_date} are below {liquidity_below}:")
            print(f"Existing Range : {liquidity_below}")
            print(filtered_closes[['Low', 'Datetime']])



