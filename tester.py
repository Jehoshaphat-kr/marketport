import tdatool as tt

if __name__ == "__main__":
    ticker = '084370'

    fundamental = tt.Fundamental(ticker=ticker)
    print(f'{fundamental.name}({fundamental.ticker}) 기본적 분석')
    print(fundamental.business_summary)

    fundamental.show_summary(show=False, save=True)
    fundamental.show_sales(show=False, save=True)
    fundamental.show_multiple(show=False, save=True)

    technical = tt.Technical(ticker=ticker, period=5)
    print(f'{technical.name}({technical.ticker}) 기술적 분석')
    technical.show_price(show=False, save=True)
    technical.show_trend(show=False, save=True)
