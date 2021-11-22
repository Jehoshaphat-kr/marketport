from charts import chart
import pandas as pd
import os


__root__ = os.path.dirname(os.path.dirname(__file__))
__meta__ = pd.read_csv(
    'http://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/market/market.csv',
    encoding='utf-8',
    index_col='종목코드'
).join(
    other=pd.read_csv(
        'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/group/WI26.csv',
        encoding='utf-8',
        index_col='종목코드'
    ).drop(columns=['종목명']),
    how='left'
)[['섹터', '종목명', '종가', '시가총액', 'R1D', 'R1W', 'R1M', 'PER', 'DIV']].rename(
    columns={'R1D': '1일등락', 'R1W': '1주등락', 'R1M': '1개월등락'}
)
for col in __meta__.columns:
    if '등락' in col:
        __meta__[col] = round(__meta__[col], 2).astype(str) + '%'
__meta__['PER'] = round(__meta__['PER'], 2)
__meta__['종가'] = __meta__['종가'].apply(lambda p: '{:,}원'.format(int(p)))
cap = (__meta__["시가총액"] / 100000000).astype(int).astype(str)
__meta__['시가총액'] = cap.apply(lambda v: v + "억" if len(v) < 5 else v[:-4] + '조 ' + v[-4:] + '억')
__meta__.index = __meta__.index.astype(str).str.zfill(6)


class Equity(chart):
    def __init__(
        self,
        ticker: str = '005930',
        data_src: str = 'online',
        filter_key: str = '종가',
        filter_win: list = None,
        sr_diverse: bool = False
    ):
        self.name = name = __meta__.loc[ticker, '종목명']
        self.filter_win = [5, 10, 20, 60, 120] if not filter_win else filter_win
        super().__init__(ticker=ticker, data_src=data_src, filter_key=filter_key)
    #
    #     self.ticker = ticker
    #     self.filter_win = [5, 10, 20, 60, 120] if not filter_win else filter_win
    #     self.sr_diverse = sr_diverse
    #
    #     self.price = pd.read_csv(
    #         f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{ticker}.csv',
    #         encoding='utf-8',
    #         index_col='날짜'
    #     ) if data_src == 'online' else pd.read_csv(
    #         os.path.join(__root__, f'warehouse/series/{ticker}.csv'),
    #         encoding='utf-8',
    #         index_col='날짜'
    #     )
    #     self.price.index = pd.to_datetime(self.price.index)
    #     self.series = self.price[filter_key]
    #
    #     self.filter = pd.concat([self.sma, self.ema, self.fir, self.iir], axis=1)
    #     return

    def show_s_price(self, save:bool=False, show:bool=False):
        return self.s_price(save=save, show=show)

if __name__ == "__main__":
    myEquity = Equity(ticker='005930')
    print(f'{myEquity.name}({myEquity.ticker}) 분석')

    myEquity.show_s_price(save=True )
    # print(myEquity.guide)