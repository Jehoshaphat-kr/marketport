import tdatool as tt

if __name__ == "__main__":
    ticker = '035420'

    # technical = tt.Technical(ticker=ticker, data_src='offline')
    # print(f'{technical.name}({technical.ticker}) 기술적 분석')
    # technical.s_price(show=False, save=True)
    # technical.s_trend(show=False, save=True)

    fundamental = tt.Fundamental(ticker=ticker)
    print(f'{fundamental.name}({fundamental.ticker}) 기본적 분석')
    fundamental.show_sales(show=False, save=True)