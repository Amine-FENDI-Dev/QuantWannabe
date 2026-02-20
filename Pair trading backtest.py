# Import libraries
import matplotlib.pyplot as plt   # used to draw graphs
import numpy as np                # used for math operations
import pandas as pd               # used to work with tables (DataFrames)
import yfinance as yf             # used to download stock market data
import statsmodels.api as sm      # used for regression and statistical tests


# ============================================================
# Engle-Granger Cointegration Test
# ============================================================
def EG_method(X, Y, show_summary=False):
    
    # STEP 1:
    # Run linear regression: try to explain Y using X
    # This estimates: Y = aX + b
    model1 = sm.OLS(Y, sm.add_constant(X)).fit()
    
    # Residuals = difference between real Y and predicted Y
    # If the two assets move together long-term,
    # these residuals should stay stable (stationary)
    epsilon = model1.resid

    if show_summary:
        print('\nStep 1\n')
        print(model1.summary())
    
    # Run Augmented Dickey-Fuller test on residuals
    # If p-value > 0.05 → residuals are NOT stationary
    # If not stationary → no cointegration
    if sm.tsa.stattools.adfuller(epsilon)[1] > 0.05:
        return False, model1
    
    # STEP 2:
    # Take first difference (daily change) of X
    # Add lagged residual (error correction term)
    X_dif = sm.add_constant(
        pd.concat([X.diff(), epsilon.shift(1)], axis=1).dropna()
    )

    # First difference of Y (daily change)
    Y_dif = Y.diff().dropna()
    
    # Run regression again:
    # ΔY = αΔX + β * previous_error
    model2 = sm.OLS(Y_dif, X_dif).fit()
    
    if show_summary:
        print('\nStep 2\n')
        print(model2.summary())
    
    # The coefficient of the error term must be negative
    # Negative means system pulls back toward equilibrium
    if list(model2.params)[-1] > 0:
        return False, model1
    else:
        return True, model1


# ============================================================
# Signal Generation
# ============================================================
def signal_generation(asset1, asset2, method, bandwidth=250):    
    
    signals = pd.DataFrame()
    
    # Store closing prices
    signals['asset1'] = asset1['Close']
    signals['asset2'] = asset2['Close']
    
    # signals1 = direction for asset1
    # 1 = long, -1 = short, 0 = no position
    signals['signals1'] = 0    
    signals['signals2'] = 0
    
    prev_status = False  # was cointegrated yesterday?
    
    # Create empty columns for statistics
    signals['z'] = np.nan
    signals['z upper limit'] = np.nan
    signals['z lower limit'] = np.nan
    signals['fitted'] = np.nan    
    signals['residual'] = np.nan
    
    # Loop through data using rolling window
    for i in range(bandwidth, len(signals)):
        
        # Test cointegration using last 250 days
        coint_status, model = method(
            signals['asset1'].iloc[i-bandwidth:i],
            signals['asset2'].iloc[i-bandwidth:i]
        )
                
        # If cointegration breaks → close positions
        if prev_status and not coint_status:
            if signals.at[signals.index[i-1], 'signals1'] != 0:
                signals.at[signals.index[i], 'signals1'] = 0
                signals.at[signals.index[i], 'signals2'] = 0
                
                # Reset statistics
                signals['z'].iloc[i:] = np.nan
                signals['z upper limit'].iloc[i:] = np.nan
                signals['z lower limit'].iloc[i:] = np.nan
                signals['fitted'].iloc[i:] = np.nan    
                signals['residual'].iloc[i:] = np.nan
        
        # If cointegration starts
        if not prev_status and coint_status:
            
            # Predict asset2 using regression model
            signals['fitted'].iloc[i:] = model.predict(
                sm.add_constant(signals['asset1'].iloc[i:])
            )
            
            # residual = real price - predicted price
            signals['residual'].iloc[i:] = (
                signals['asset2'].iloc[i:] - signals['fitted'].iloc[i:]
            )
            
            # Compute z-score:
            # z = (value - mean) / standard deviation
            signals['z'].iloc[i:] = (
                (signals['residual'].iloc[i:] - np.mean(model.resid)) /
                np.std(model.resid)
            )
            
            # Set upper and lower bounds (1 standard deviation)
            signals['z upper limit'].iloc[i:] = (
                signals['z'].iloc[i] + np.std(model.resid)
            )
            signals['z lower limit'].iloc[i:] = (
                signals['z'].iloc[i] - np.std(model.resid)
            )
        
        # Trading rule:
        # If z-score too high → spread likely to fall
        if coint_status and signals['z'].iloc[i] > signals['z upper limit'].iloc[i]:
            signals.at[signals.index[i], 'signals1'] = 1
            
        # If z-score too low → spread likely to rise
        if coint_status and signals['z'].iloc[i] < signals['z lower limit'].iloc[i]:
            signals.at[signals.index[i], 'signals1'] = -1
                
        prev_status = coint_status    
    
    # Convert holding signals into actual trade signals
    signals['positions1'] = signals['signals1'].diff()
    
    # Second asset takes opposite direction
    signals['signals2'] = -signals['signals1']
    signals['positions2'] = signals['signals2'].diff()   
    
    return signals


# ============================================================
# Plot Positions
# ============================================================
def plot(data, ticker1, ticker2):    
   
    fig = plt.figure(figsize=(10,5))
    bx = fig.add_subplot(111)   
    bx2 = bx.twinx()
    
    # Plot prices
    bx.plot(data.index, data['asset1'], c='#113aac', alpha=0.7)
    bx2.plot(data.index, data['asset2'], c='#907163', alpha=0.7)

    # Plot entry points
    bx.plot(data.loc[data['positions1']==1].index,
            data['asset1'][data['positions1']==1],
            lw=0, marker='^', markersize=8, c='g')
    
    bx.plot(data.loc[data['positions1']==-1].index,
            data['asset1'][data['positions1']==-1],
            lw=0, marker='v', markersize=8, c='r')

    plt.title('Pair Trading')
    plt.grid(True)
    plt.show()


# ============================================================
# Portfolio Performance
# ============================================================
def portfolio(data):

    capital0 = 20000  # starting money

    # Number of shares we can afford
    positions1 = capital0 // max(data['asset1'])
    positions2 = capital0 // max(data['asset2'])

    data['cumsum1'] = data['positions1'].cumsum()

    portfolio = pd.DataFrame()
    
    # Value of shares held
    portfolio['holdings1'] = data['cumsum1'] * data['asset1'] * positions1
    
    # Cash left
    portfolio['cash1'] = capital0 - (
        data['positions1'] * data['asset1'] * positions1
    ).cumsum()
    
    # Total value
    portfolio['total asset1'] = portfolio['holdings1'] + portfolio['cash1']
    
    return portfolio


# ============================================================
# Main Function
# ============================================================
def main():
    
    stdate = '2013-01-01'
    eddate = '2014-12-31'
    
    ticker1 = 'NVDA'
    ticker2 = 'AMD'

    # Download historical data
    asset1 = yf.download(ticker1, start=stdate, end=eddate)
    asset2 = yf.download(ticker2, start=stdate, end=eddate)

    # Generate trading signals
    signals = signal_generation(asset1, asset2, EG_method)

    # First valid trading date
    ind = signals['z'].dropna().index[0]

    # Plot trades
    plot(signals[ind:], ticker1, ticker2)

    # Compute portfolio performance
    portfolio_details = portfolio(signals[ind:])


if __name__ == '__main__':
    main()
