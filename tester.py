import tdatool as tt

if __name__ == "__main__":
    ticker = '100090'

    stock = tt.Evaluate(ticker=ticker, src='local', period=5)
    print(f'{stock.name}({stock.ticker}) 분석')

    stock.show_business(show=True, save=True)
    stock.show_summary(show=False, save=True)
    stock.show_sales(show=False, save=True)
    stock.show_multiple(show=False, save=True)

    stock.show_price(show=False, save=True)
    stock.show_trend(show=False, save=True)
