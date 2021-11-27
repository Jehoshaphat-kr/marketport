import tdatool as tt

if __name__ == "__main__":
    ticker = '000660'

    technical = tt.Technical(ticker=ticker, period=5)
    print(f'{technical.name}({technical.ticker}) 기술적 분석')
    technical.s_price(show=False, save=True)
    technical.s_trend(show=False, save=True)

    fundamental = tt.Fundamental(ticker=ticker)
    print(f'{fundamental.name}({fundamental.ticker}) 기본적 분석')

    fundamental.show_sales(show=False, save=True)