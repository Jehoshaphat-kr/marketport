import tdatool as tt
import pandas as pd
import os


class Equity:

    # def __init__(
    #     self,
    #     ticker: str = '005930',
    #     data_src: str = 'online',
    #     filter_key: str = '종가',
    #     filter_win: list = None,
    #     sr_diverse: bool = False
    # ):
    #     self.ticker = ticker
    #     self.name = name = __meta__.loc[ticker, '종목명']
    #     self.filter_win = [5, 10, 20, 60, 120] if not filter_win else filter_win
        # super().__init__(ticker=ticker, data_src=data_src, filter_key=filter_key)
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