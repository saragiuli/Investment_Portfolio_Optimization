import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
pd.options.display.float_format = "{:.4f}".format

# ==================== IMPROVED FUNCTIONS ====================

def give_weights(n_assets):
    """Generate normalized random weights for n_assets"""
    rand = np.random.random(n_assets)
    return rand / rand.sum()

def efficient_frontier(ret_df, n_portfolios=5000, risk_free_rate=0.02):
    """
    Calculate efficient frontier with random portfolios
    
    Returns:
        DataFrame with weights, returns, risk and Sharpe ratio
    """
    n_assets = len(ret_df.columns)
    results = {
        'weights': [],
        'returns': [],
        'risk': [],
        'sharpe': []
    }
    
    for _ in range(n_portfolios):
        w = give_weights(n_assets)
        
        # Annualized return
        ret = w.dot(ret_df.mean()) * 252
        
        # Annualized risk
        risk = np.sqrt(w.dot(ret_df.cov().dot(w)) * 252)
        
        # Sharpe Ratio
        sharpe = (ret - risk_free_rate) / risk
        
        results['weights'].append(w)
        results['returns'].append(ret)
        results['risk'].append(risk)
        results['sharpe'].append(sharpe)
    
    return pd.DataFrame(results)

def calculate_max_drawdown(returns):
    """Calculate Maximum Drawdown"""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino Ratio (penalizes only downside volatility)"""
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    
    if downside_std == 0:
        return np.nan
    
    return (returns.mean() * 252 - risk_free_rate) / downside_std

def portfolio_metrics(returns, name="Portfolio"):
    """Calculate all performance metrics"""
    metrics = {
        'Name': name,
        'Annual Return': returns.mean() * 252,
        'Annual Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)),
        'Sortino Ratio': calculate_sortino_ratio(returns),
        'Max Drawdown': calculate_max_drawdown(returns),
        'Total Return': (1 + returns).prod() - 1
    }
    return metrics

def create_ml_features(df, lags=5):
    """
    Create advanced features for machine learning
    
    Features:
    - Lagged returns (1 to n days)
    - Moving averages
    - Rolling volatility
    - RSI (Relative Strength Index)
    """
    df = df.copy()
    
    # Lagged returns
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['ret'].shift(i)
    
    # Moving averages
    df['ma_5'] = df['ret'].rolling(window=5).mean()
    df['ma_20'] = df['ret'].rolling(window=20).mean()
    
    # Rolling volatility
    df['vol_5'] = df['ret'].rolling(window=5).std()
    df['vol_20'] = df['ret'].rolling(window=20).std()
    
    # Simplified RSI
    delta = df['ret']
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

# ==================== PART 1: PORTFOLIO ANALYSIS ====================

print("="*80)
print("PORTFOLIO ANALYSIS AND MACHINE LEARNING")
print("="*80)

# Download data
tickers = ["AAPL", "MMM", "KO"]
print(f"\n Downloading data for: {', '.join(tickers)}")

df = yf.download(tickers, start='2015-01-01', auto_adjust=False, progress=False)['Adj Close']
print(f"✓ Data downloaded: {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")

# Calculate returns
ret_df = df.pct_change().dropna()

print("\n" + "="*80)
print("1. INDIVIDUAL STOCK ANALYSIS")
print("="*80)

# Metrics for each stock
for ticker in tickers:
    metrics = portfolio_metrics(ret_df[ticker], ticker)
    print(f"\n{ticker}:")
    for key, value in metrics.items():
        if key != 'Name':
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Cumulative returns chart
cum_ret = (ret_df + 1).cumprod() - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

cum_ret.plot(ax=ax1, linewidth=2)
ax1.set_title("Cumulative Returns of Individual Stocks (since 2015)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Cumulative Return")
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Correlation matrix
sns.heatmap(ret_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax2, 
            fmt='.3f', square=True, linewidths=1)
ax2.set_title("Correlation Matrix between Stocks", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/stock_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== PART 2: EFFICIENT FRONTIER ====================

print("\n" + "="*80)
print("2. EFFICIENT FRONTIER AND OPTIMIZATION")
print("="*80)

print("\n🔄 Computing 5000 random portfolios...")
portfolios = efficient_frontier(ret_df, n_portfolios=5000)

# Find optimal portfolio (max Sharpe)
max_sharpe_idx = portfolios['sharpe'].idxmax()
max_sharpe_portfolio = portfolios.loc[max_sharpe_idx]

# Find minimum variance portfolio
min_var_idx = portfolios['risk'].idxmin()
min_var_portfolio = portfolios.loc[min_var_idx]

print(f"\n✓ Portfolio with Max Sharpe Ratio:")
print(f"  Annual Return: {max_sharpe_portfolio['returns']:.4f}")
print(f"  Annual Risk: {max_sharpe_portfolio['risk']:.4f}")
print(f"  Sharpe Ratio: {max_sharpe_portfolio['sharpe']:.4f}")
print(f"  Weights:")
for i, ticker in enumerate(tickers):
    print(f"    {ticker}: {max_sharpe_portfolio['weights'][i]:.2%}")

print(f"\n✓ Minimum Variance Portfolio:")
print(f"  Annual Return: {min_var_portfolio['returns']:.4f}")
print(f"  Annual Risk: {min_var_portfolio['risk']:.4f}")
print(f"  Sharpe Ratio: {min_var_portfolio['sharpe']:.4f}")
print(f"  Weights:")
for i, ticker in enumerate(tickers):
    print(f"    {ticker}: {min_var_portfolio['weights'][i]:.2%}")

# Efficient Frontier plot
plt.figure(figsize=(14, 8))
scatter = plt.scatter(portfolios['risk'], portfolios['returns'], 
                     c=portfolios['sharpe'], cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Sharpe Ratio')

# Highlight special portfolios
plt.scatter(max_sharpe_portfolio['risk'], max_sharpe_portfolio['returns'], 
           color='red', s=200, marker='*', edgecolors='black', linewidths=2,
           label='Max Sharpe Ratio', zorder=5)
plt.scatter(min_var_portfolio['risk'], min_var_portfolio['returns'], 
           color='yellow', s=200, marker='*', edgecolors='black', linewidths=2,
           label='Min Variance', zorder=5)

plt.xlabel('Annualized Risk (Volatility)', fontsize=12)
plt.ylabel('Annualized Return', fontsize=12)
plt.title('Efficient Frontier - 5000 Random Portfolios', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/efficient_frontier.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== PART 3: MACHINE LEARNING ====================

print("\n" + "="*80)
print("3. MACHINE LEARNING - MARKET DIRECTION PREDICTION")
print("="*80)

# Create equally-weighted portfolio
ptf_returns = ret_df.mean(axis=1)
ptf_cum = (ptf_returns + 1).cumprod() - 1

# Prepare data for ML
ml_df = pd.DataFrame({
    'cum_ret': ptf_cum,
    'ret': ptf_returns
})

# Create target: market direction (1 = up, 0 = down)
ml_df['direction'] = np.where(ml_df['ret'] > 0, 1, 0)

# Create advanced features
print("\n🔧 Creating features for ML...")
ml_df = create_ml_features(ml_df, lags=5)
ml_df.dropna(inplace=True)

# Select features
feature_cols = [col for col in ml_df.columns if col not in ['cum_ret', 'ret', 'direction']]
X = ml_df[feature_cols]
y = ml_df['direction']

print(f"✓ Features created: {len(feature_cols)}")
print(f"  {', '.join(feature_cols[:5])}...")
print(f"✓ Total samples: {len(X)}")
print(f"  Class 1 (Up): {y.sum()} ({y.sum()/len(y):.1%})")
print(f"  Class 0 (Down): {len(y)-y.sum()} ({(len(y)-y.sum())/len(y):.1%})")

# Train/test split (70/30 without shuffle to respect temporal order)
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n📊 Train set: {len(X_train)} samples")
print(f"📊 Test set: {len(X_test)} samples")

# ==================== MODEL 1: LOGISTIC REGRESSION ====================

print("\n" + "-"*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("-"*80)

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train, y_train)

# Predictions
lr_pred_train = lr_model.predict(X_train)
lr_pred_test = lr_model.predict(X_test)

# Accuracy
print(f"\nTrain Accuracy: {accuracy_score(y_train, lr_pred_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, lr_pred_test):.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, lr_pred_test, target_names=['Down', 'Up']))

# ==================== MODEL 2: RANDOM FOREST ====================

print("\n" + "-"*80)
print("MODEL 2: RANDOM FOREST")
print("-"*80)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                  class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

# Accuracy
print(f"\nTrain Accuracy: {accuracy_score(y_train, rf_pred_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, rf_pred_test):.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, rf_pred_test, target_names=['Down', 'Up']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# ==================== BACKTESTING STRATEGY ====================

print("\n" + "="*80)
print("4. TRADING STRATEGY BACKTESTING")
print("="*80)

# Prepare dataframe for backtesting
backtest_df = pd.DataFrame(index=X_test.index)
backtest_df['ret'] = ml_df.loc[X_test.index, 'ret']
backtest_df['lr_signal'] = lr_pred_test
backtest_df['rf_signal'] = rf_pred_test

# Strategy: invest only when model predicts UP (1)
backtest_df['lr_strategy'] = backtest_df['lr_signal'] * backtest_df['ret']
backtest_df['rf_strategy'] = backtest_df['rf_signal'] * backtest_df['ret']

# Cumulative returns
backtest_df['buy_hold'] = (1 + backtest_df['ret']).cumprod() - 1
backtest_df['lr_cum'] = (1 + backtest_df['lr_strategy']).cumprod() - 1
backtest_df['rf_cum'] = (1 + backtest_df['rf_strategy']).cumprod() - 1

# Calculate metrics
strategies = {
    'Buy & Hold': backtest_df['ret'],
    'Logistic Regression': backtest_df['lr_strategy'],
    'Random Forest': backtest_df['rf_strategy']
}

print("\nPERFORMANCE METRICS:")
print("-"*80)

metrics_comparison = []
for name, returns in strategies.items():
    metrics = portfolio_metrics(returns, name)
    metrics_comparison.append(metrics)
    
    print(f"\n{name}:")
    for key, value in metrics.items():
        if key != 'Name':
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Convert to DataFrame for visualization
metrics_df = pd.DataFrame(metrics_comparison).set_index('Name')

# ==================== FINAL VISUALIZATIONS ====================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Strategy comparison
backtest_df[['buy_hold', 'lr_cum', 'rf_cum']].plot(ax=axes[0, 0], linewidth=2)
axes[0, 0].set_title('Strategy Comparison - Cumulative Returns', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Cumulative Return')
axes[0, 0].legend(['Buy & Hold', 'Logistic Regression', 'Random Forest'])
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix - Logistic Regression
cm_lr = confusion_matrix(y_test, lr_pred_test)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix - Logistic Regression', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('True')
axes[0, 1].set_xlabel('Predicted')

# 3. Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, rf_pred_test)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix - Random Forest', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('True')
axes[1, 0].set_xlabel('Predicted')

# 4. Feature Importance
feature_importance.head(10).plot(x='feature', y='importance', kind='barh', ax=axes[1, 1])
axes[1, 1].set_title('Top 10 Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('images/ml_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Final chart with drawdown
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Cumulative returns
backtest_df[['buy_hold', 'lr_cum', 'rf_cum']].plot(ax=ax1, linewidth=2)
ax1.set_title('Strategy Performance Over Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Return')
ax1.legend(['Buy & Hold', 'Logistic Regression', 'Random Forest'], loc='best')
ax1.grid(True, alpha=0.3)

# Drawdown
for col, label in [('ret', 'Buy & Hold'), ('lr_strategy', 'LR'), ('rf_strategy', 'RF')]:
    cum = (1 + backtest_df[col]).cumprod()
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max
    ax2.plot(drawdown.index, drawdown.values, label=label, linewidth=2)

ax2.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
ax2.set_ylabel('Drawdown')
ax2.set_xlabel('Date')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.fill_between(backtest_df.index, 0, 
                 [(1 + backtest_df['ret']).cumprod()[i] / 
                  (1 + backtest_df['ret']).cumprod()[:i+1].max() - 1 
                  for i in range(len(backtest_df))], alpha=0.3)

plt.tight_layout()
plt.savefig('images/performance_drawdown.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETED!")
print("="*80)
print("\nFiles saved in images/ folder:")
print("  - stock_analysis.png")
print("  - efficient_frontier.png")
print("  - ml_results.png")
print("  - performance_drawdown.png")