import yfinance as yf
import pandas as pd
import numpy as numpy
from smartmoneyconcepts import smc as smartmoneyconcept
from custom_smc_indicator import smc
# import smartmoney
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

us_stocks_data = pd.read_csv("US_30_Stocks.csv")



main_pd_array = pd.DataFrame()

# Define the end date for filtering
# end_date = "2025-01-28"
end_date = datetime.today().strftime('%Y-%m-%d')

for idx in range(len(us_stocks_data)):
    ticker = us_stocks_data.loc[idx, 'Symbol']
    # # Define the ticker symbol
    # ticker = "NKE"
    print(ticker)
    # Download recent minute-level data to get the latest closing price
    minute_data = yf.download(ticker, interval="1m", period="5d")

    # Filter data to include only dates before the specified end date
    minute_data = minute_data[minute_data.index < end_date]

    # Get the latest closing price
    current_closing_price = minute_data['Close'].iloc[-1] if not minute_data.empty else None
    print(f"Current Closing Price: {current_closing_price}")

    # Define intervals and periods
    intervals = {"monthly": "1mo", "weekly": "1wk", "daily": "1d", "hourly": "1h","15min":"15m","5min":"5m"}
    periods = {"monthly": "6y", "weekly": "5y", "daily": "2y", "hourly": "60d","15min":"30d","5min":"10d"}


    results = []

    for timeframe, interval in intervals.items():
        # Download data for each timeframe
        data = yf.download(ticker, interval=interval, period=periods[timeframe])

        data = data[data.index < end_date]

        # Prepare the OHLC dataframe
        ohlc = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlc.reset_index(inplace=True)
        if interval in ['1h', '15m', '5m']:
            ohlc['datetime'] = pd.to_datetime(ohlc['Datetime'])
        else:
            ohlc['datetime'] = pd.to_datetime(ohlc['Date'])
        ohlc.set_index('datetime', inplace=True)
        ohlc = ohlc[['open', 'high', 'low', 'close', 'volume']]

        # Apply SMC Fair Value Gap (FVG) detection
        fvg_data = smartmoneyconcept.fvg(ohlc, join_consecutive=False)
        fvg_not_nan_filtered = fvg_data[fvg_data['FVG'].notna()]

        # Filter for levels above and below the current price
        fvg_above_df = fvg_not_nan_filtered[(fvg_not_nan_filtered['MitigatedIndex'] == 0) & (fvg_not_nan_filtered["Bottom"] > current_closing_price)].sort_values(by="Bottom")
        fvg_below_df = fvg_not_nan_filtered[(fvg_not_nan_filtered['MitigatedIndex'] == 0) & (fvg_not_nan_filtered["Top"] < current_closing_price)].sort_values(by="Top", ascending=False)

        fvg_above_top = round(fvg_above_df.iloc[0]["Top"],2) if not fvg_above_df.empty else None
        fvg_above_bottom = round(fvg_above_df.iloc[0]["Bottom"],2) if not fvg_above_df.empty else None
        fvg_below_top = round(fvg_below_df.iloc[0]["Top"],2) if not fvg_below_df.empty else None
        fvg_below_bottom = round(fvg_below_df.iloc[0]["Bottom"],2) if not fvg_below_df.empty else None

        fvg_above_date = ohlc.iloc[fvg_above_df.index[0]].name.strftime('%Y-%m-%d') if not fvg_above_df.empty else None
        fvg_below_date = ohlc.iloc[fvg_below_df.index[0]].name.strftime('%Y-%m-%d') if not fvg_below_df.empty else None

        # Apply SMC Swing Highs and Lows
        swing_highs_lows = smartmoneyconcept.swing_highs_lows(ohlc, swing_length=4)
        swing_highs_lows_not_nan_filtered = swing_highs_lows[swing_highs_lows['HighLow'].notna()]

        # Get the index to exclude
        exclude_index = len(ohlc) - 1

        # Exclude the row with the index
        swing_highs_lows_not_nan_filtered = swing_highs_lows_not_nan_filtered.drop(index=exclude_index, errors='ignore')

        swing_highs_lows_above_df = swing_highs_lows_not_nan_filtered[(swing_highs_lows_not_nan_filtered["Level"] > current_closing_price) & (swing_highs_lows_not_nan_filtered["HighLow"] == 1)].sort_index(ascending=False)
        swing_highs_lows_below_df = swing_highs_lows_not_nan_filtered[(swing_highs_lows_not_nan_filtered["Level"] < current_closing_price) & (swing_highs_lows_not_nan_filtered["HighLow"] == -1)].sort_index(ascending=False)

        swing_highs_lows_above_top = round(swing_highs_lows_above_df.iloc[0]["Level"],2) if not swing_highs_lows_above_df.empty else None
        swing_highs_lows_below_bottom = round(swing_highs_lows_below_df.iloc[0]["Level"],2) if not swing_highs_lows_below_df.empty else None

        swing_highs_lows_above_date = ohlc.iloc[swing_highs_lows_above_df.index[0]].name.strftime('%Y-%m-%d') if not swing_highs_lows_above_df.empty else None
        swing_highs_lows_below_date = ohlc.iloc[swing_highs_lows_below_df.index[0]].name.strftime('%Y-%m-%d') if not swing_highs_lows_below_df.empty else None


        # Apply SMC Swing Highs and Lows
        swing_highs_lows = smartmoneyconcept.swing_highs_lows(ohlc, swing_length=10)
        # Apply SMC Order Blocks (OB)
        ob_data = smartmoneyconcept.ob(ohlc, swing_highs_lows, close_mitigation=False)
        filtered_ob_index = ob_data.dropna(subset=["OB"])

        breaker_ob = filtered_ob_index[filtered_ob_index['MitigatedIndex'] != 0]

        ohlc_filtered_ob = ohlc.iloc[filtered_ob_index.index]
        ohlc_filtered_ob.reset_index(inplace=True, drop=False)

        ohlc_filtered_breaker_ob = ohlc.iloc[breaker_ob.index]
        ohlc_filtered_breaker_ob.reset_index(inplace=True, drop=False)


        ob_above_df = ohlc_filtered_ob[ohlc_filtered_ob["high"] > current_closing_price].sort_values(by="high")
        ob_below_df = ohlc_filtered_ob[ohlc_filtered_ob["low"] < current_closing_price].sort_values(by="low", ascending=False)
        breaker_ob_above_df = ohlc_filtered_breaker_ob[ohlc_filtered_breaker_ob["high"] > current_closing_price].sort_values(by="high")
        breaker_ob_below_df = ohlc_filtered_breaker_ob[ohlc_filtered_breaker_ob["low"] < current_closing_price].sort_values(by="low", ascending=False)

        ob_above_top = round(ob_above_df.iloc[0]["high"],2) if not ob_above_df.empty else None
        ob_above_bottom = round(ob_above_df.iloc[0]["low"],2) if not ob_above_df.empty else None
        ob_below_top = round(ob_below_df.iloc[0]["high"],2) if not ob_below_df.empty else None
        ob_below_bottom = round(ob_below_df.iloc[0]["low"],2) if not ob_below_df.empty else None

        breaker_ob_above_top = round(breaker_ob_above_df.iloc[0]["high"],2) if not breaker_ob_above_df.empty else None
        breaker_ob_above_bottom = round(breaker_ob_above_df.iloc[0]["low"],2) if not breaker_ob_above_df.empty else None
        breaker_ob_below_top = round(breaker_ob_below_df.iloc[0]["high"],2) if not breaker_ob_below_df.empty else None
        breaker_ob_below_bottom = round(breaker_ob_below_df.iloc[0]["low"],2) if not breaker_ob_below_df.empty else None

        # ob_above_date = ohlc.iloc[ob_above_df.index[0]].name.strftime('%Y-%m-%d') if not ob_above_df.empty else None
        # ob_below_date = ohlc.iloc[ob_below_df.index[0]].name.strftime('%Y-%m-%d') if not ob_below_df.empty else None

        ob_above_date = ob_above_df.iloc[0]['datetime'].strftime('%Y-%m-%d') if not ob_above_df.empty else None
        ob_below_date = ob_below_df.iloc[0]['datetime'].strftime('%Y-%m-%d') if not ob_below_df.empty else None

        breaker_ob_above_date = breaker_ob_above_df.iloc[0]['datetime'].strftime('%Y-%m-%d') if not breaker_ob_above_df.empty else None
        breaker_ob_below_date = breaker_ob_below_df.iloc[0]['datetime'].strftime('%Y-%m-%d') if not breaker_ob_below_df.empty else None

        # Apply SMC Swing Highs and Lows
        swing_highs_lows = smc.smc.swing_highs_lows(ohlc, swing_length=3)

        bos_choch_data = smartmoneyconcept.bos_choch(ohlc, swing_highs_lows)
        # Filter rows where 'BOS' column is NOT NaN
        bos_choch_not_nan_filtered = bos_choch_data[bos_choch_data['BrokenIndex'].notna()]

        # Filter for levels above and below the current price
        bos_choch_above_df = bos_choch_not_nan_filtered[(bos_choch_not_nan_filtered["Level"] > current_closing_price) & (bos_choch_not_nan_filtered["BOS"] == 1)].sort_values(by="Level")
        bos_choch_below_df = bos_choch_not_nan_filtered[(bos_choch_not_nan_filtered["Level"] < current_closing_price) & (bos_choch_not_nan_filtered["BOS"] == -1)].sort_values(by="Level", ascending=False)

        bos_choch_above = bos_choch_above_df.iloc[0]["Level"] if not bos_choch_above_df.empty else None
        bos_choch_below = bos_choch_below_df.iloc[0]["Level"] if not bos_choch_below_df.empty else None

        bos_choch_above_formed = ohlc.iloc[bos_choch_below_df.index[0]].name.strftime('%Y-%m-%d') if not bos_choch_below_df.empty else None
        bos_choch_above_broken = ohlc.iloc[int(bos_choch_below_df.iloc[0]["BrokenIndex"])].name.strftime('%Y-%m-%d') if not bos_choch_below_df.empty else None


        # Apply SMC Swing Highs and Lows
        swing_highs_lows = smc.smc.swing_highs_lows(ohlc, swing_length=3)
        liquidity_df = smartmoneyconcept.liquidity(ohlc, swing_highs_lows)
        liquidity_not_nan_filtered = liquidity_df[liquidity_df['Liquidity'].notna()]

        filter_liquidity_untouched = liquidity_not_nan_filtered[liquidity_not_nan_filtered['Swept'] == 0]

        # Filter for levels above and below the current price
        liquidity_untouched_above_df = filter_liquidity_untouched[(filter_liquidity_untouched["Level"] > current_closing_price)].sort_values(by="Level")
        liquidity_untouched_below_df = filter_liquidity_untouched[(filter_liquidity_untouched["Level"] < current_closing_price)].sort_values(by="Level", ascending=False)

        liquidity_untouched_above = liquidity_untouched_above_df.iloc[0]["Level"] if not liquidity_untouched_above_df.empty else None
        liquidity_untouched_below = liquidity_untouched_below_df.iloc[0]["Level"] if not liquidity_untouched_below_df.empty else None

        liquidity_untouched_above_date = ohlc.iloc[liquidity_untouched_above_df.index[0]].name.strftime('%Y-%m-%d') if not liquidity_untouched_above_df.empty else None
        liquidity_untouched_below_date = ohlc.iloc[liquidity_untouched_below_df.index[0]].name.strftime('%Y-%m-%d') if not liquidity_untouched_below_df.empty else None

        # Compile results
        results.append({
            "Timeframe": timeframe,
            "FVG_Above_Top": fvg_above_top,
            "FVG_Above_Bottom": fvg_above_bottom,
            "FVG_Above_Date":fvg_above_date,
            "FVG_Below_Top": fvg_below_top,
            "FVG_Below_Bottom": fvg_below_bottom,
            "FVG_Below_Date":fvg_below_date,
            "Swing_High": swing_highs_lows_above_top,
            "Swing_High_Date":swing_highs_lows_above_date,
            "Swing_Low": swing_highs_lows_below_bottom,
            "Swing_Low_Date":swing_highs_lows_below_date,
            "OB_Above_Top": ob_above_top,
            "OB_Above_Bottom": ob_above_bottom,
            "OB_Above_Date":ob_above_date,
            "OB_Below_Top": ob_below_top,
            "OB_Below_Bottom": ob_below_bottom,
            "OB_Below_Date":ob_below_date,
            "Breaker_OB_Above_Top": breaker_ob_above_top,
            "Breaker_OB_Above_Bottom": breaker_ob_above_bottom,
            "Breaker_OB_Above_Date":breaker_ob_above_date,
            "Breaker_OB_Below_Top": breaker_ob_below_top,
            "Breaker_OB_Below_Bottom": breaker_ob_below_bottom,
            "Breaker_OB_Below_Date":breaker_ob_below_date,
            "BOS_CHOCH_Above": bos_choch_above,
            "BOS_CHOCH_Below": bos_choch_below,
            "BOS_CHOCH_Above_Formed_Date":bos_choch_above_formed,
            "BOS_CHOCH_Below_Broken_Date":bos_choch_above_broken,
            "Liquidity_untouched_Above": liquidity_untouched_above,
            "Liquidity_untouched_Below": liquidity_untouched_below,
            "Liquidity_untouched_Above_Date": liquidity_untouched_above_date,
            "Liquidity_untouched_Below_Date": liquidity_untouched_below_date
        })

    # Create the final DataFrame
    result_df = pd.DataFrame(results)
    print(result_df)

    # Transform dataframe
    single_row = {}
    for _, row in result_df.iterrows():
        prefix = f"{row['Timeframe']}_"
        for col in result_df.columns[1:]:
            single_row[f"{prefix}{col}"] = row[col]

    # Create the single-row dataframe
    transformed_df = pd.DataFrame([single_row])

    transformed_df['Stock'] = ticker

    transformed_df['Current_Price'] = current_closing_price

    # Append the data to the main DataFrame
    main_pd_array = pd.concat([main_pd_array, transformed_df], ignore_index=True)


main_pd_array.to_csv('PD_Array_Data.csv')