# 🧠 Optimal Portfolio Optimization with Machine Learning

This project focuses on constructing an optimal financial portfolio by selecting the most efficient set of assets in terms of risk-return trade-off. It uses modern portfolio theory principles along with machine learning models to evaluate and backtest different asset combinations.

---

## 📈 Objectives

- Select assets to build an efficient portfolio
- Optimize portfolio based on risk-return analysis
- Apply machine learning algorithms for performance evaluation and backtesting
- Visualize efficient frontier and portfolio performance metrics

---

## 📁 Project Structure

```
├── opt_portfolio.py              # Main script where it's store and computerd the optimization process
├── yfinance_change:2025.py              # Code that remember the changes of the last year (main in pandas visualization)
├── images/                 # Visualization outputs and results
│   ├── efficient_frontier.png
│   ├── ml_results.png
│   ├── performance_drawdown.png
│   └── stock_analysis.png
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.11+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saragiuli/Investment_Portfolio_Optimization.git
   cd Investment_Portfolio_Optimization
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```


### Running the Project

1. **Run the main analysis script**
   ```bash
   python opt_portfolio.py
   ```

3. **View results**: Check the `images/` folder for generated visualizations

---

## 📊 Features

- **Asset Selection**: Automated selection of assets based on historical performance
- **Risk-Return Analysis**: Calculate Sharpe ratio, volatility, and expected returns
- **Efficient Frontier**: Visualize optimal portfolios across different risk levels
- **ML Models**: Implement predictive models for asset performance
- **Backtesting**: Test portfolio strategies on historical data

---

## ⚙️ Technologies Used

- **Python** 🐍 - Core programming language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib / Plotly** - Data visualization
- **YFinance** - Financial data retrieval

---

## 📸 Sample Results

[Aggiungi qui screenshot delle tue visualizzazioni]

![Portfolio Efficient Frontier](images/efficient_frontier.png)
![Risk-Return Analysis](images/ml_results.png)
![stock_analysis](images/stock_analysis.png)

---

## 🤝 Contributing

Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---


## 👤 Author

**[Il tuo nome]**
- GitHub: [@saragiuli](https://github.com/saragiuli)

---

## 📚 References

- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*
