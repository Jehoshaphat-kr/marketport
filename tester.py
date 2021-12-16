import tdatool as tt

if __name__ == "__main__":
    # ticker = '005930' # 삼성전자
    # ticker = '006400' # 삼성SDI
    # ticker = '009150' # 삼성전기
    # ticker = '000660' # SK하이닉스
    # ticker = '035720' # 카카오
    # ticker = '035420' # NAVER
    # ticker = '247540' # 에코프로비엠
    # ticker = '207940' # 삼성바이오로직스
    # ticker = '005380' # 현대차
    # ticker = '243840' # 신흥에스이씨
    ticker = '007070'

    meta = tt.metadata().market_data

    stock = tt.analytic(ticker=ticker, src='local', period=5, meta=meta)
    print(f'{stock.name}({stock.ticker}) 분석')
    print('=' * 120)
    print(stock.summary)

    # stock.show_overview(show=True, save=False)
    # stock.show_supply(show=False, save=True)
    # stock.show_multiples(show=False, save=True)
    # stock.show_basic(show=False, save=True)
    # stock.show_bollinger(show=False, save=True)
    # stock.show_rsi(show=False, save=True)
    # stock.show_cost(show=False, save=True)
    # stock.show_momentum(show=False, save=True)
    # stock.show_overtrade(show=False, save=True)
    # stock.show_vortex(show=True, save=False)