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
    ticker = '005380' # 현대차
    # ticker = '243840' # 신흥에스이씨

    stock = tt.Evaluate(ticker=ticker, src='local', period=5)
    print(f'{stock.name}({stock.ticker}) 분석')
    print('=' * 120)
    print(stock.summary)

    chart = tt.Chart(obj=stock)

    # chart.show_overview(show=False, save=True)
    # chart.show_supply(show=False, save=True)
    # chart.show_multiples(show=False, save=True)
    # chart.show_basic(show=False, save=True)
    # chart.show_bollinger(show=False, save=True)
    # chart.show_rsi(show=False, save=True)
    # chart.show_cost(show=False, save=True)