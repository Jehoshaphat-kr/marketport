import tdatool as tt

if __name__ == "__main__":
    ticker = '006400'

    stock = tt.Evaluate(ticker=ticker, src='local', period=5)
    print(f'{stock.name}({stock.ticker}) 분석')

    chart = tt.Chart(obj=stock)

    # stock.show_business(show=True, save=True)
    # stock.show_summary(show=False, save=True)
    # stock.show_sales(show=False, save=True)
    # stock.show_multiple(show=False, save=True)

    chart.show_basic(show=False, save=True)
    chart.show_trend(show=False, save=True)
    chart.show_bollinger(show=False, save=True)
