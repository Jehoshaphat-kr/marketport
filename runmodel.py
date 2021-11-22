import tdatool as tt

if __name__ == "__main__":
    myEquity = tt.TimeSeries(
        ticker='011790',
        data_src='offline'
    )
    print(f'{myEquity.name}({myEquity.ticker}) 기본 분석')
    myEquity.s_price(show=False, save=True)
    myEquity.s_trend(show=False, save=True)
    # print(tb.meta)