from datetime import datetime
from pykrx import stock
import pandas as pd
import numpy as np
import os


__root__ = os.path.dirname(os.path.dirname(__file__))
warehouse = os.path.join(__root__, 'warehouse')
class statistic:

    dir_storage = os.path.join(warehouse, 'series')
    frame = pd.DataFrame()

    def is_trade_banned(self, ticker) -> bool:
        self.frame = pd.read_csv(
            filepath_or_buffer=os.path.join(self.dir_storage, f'{ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        self.frame.drop(columns=['거래량'], inplace=True)

        today_price = self.frame.iloc[-1].values
        if today_price[0] == today_price[1] == 0:
            return True
        return False

    @property
    def performance(self) -> list:
        """
        기간별 수익률
        :return:
        """
        price = self.frame['종가']
        return [100 * price.pct_change(periods=ago).values[-1] for ago in [1, 5, 21, 63, 126, 252]]


class market(statistic):

    def __init__(self, date:datetime=None):
        print("=" * 50)
        print("|" + " " * 14 + "시장 데이터 업데이트" + " " * 14 + "|")
        print("=" * 50)
        self.base = pd.read_csv(
            filepath_or_buffer=os.path.join(warehouse, 'meta-stock.csv'),
            encoding='utf-8',
            index_col='종목코드'
        )
        self.base.index = self.base.index.astype(str).str.zfill(6)
        self.base.drop(columns=['상장일'], inplace=True)
        self.today = date if date else datetime.today()
        return

    def update_percentage(self) -> None:
        """
        기간별 수익률 산출
        :return:
        """
        print("Proc 01: 시장 수익률 취합 중...")
        perf = []
        risk = []
        index = []
        for ticker in self.base.index:
            if self.is_trade_banned(ticker=ticker):
                continue
            perf.append(self.performance)
            index.append(ticker)

        df_perf = pd.DataFrame(
            data=perf,
            index=index,
            columns=['R1D', 'R1W', 'R1M', 'R3M', 'R6M', 'R1Y'],
        )
        self.base = pd.concat(objs=[self.base, df_perf], axis=1)
        self.base.index.name = '종목코드'
        return

    def update_multiple(self) -> None:
        """
        배수 다운로드 및 업데이트
        :return:
        """
        print("Proc 02: 시장 배수 다운로드 중...")
        fetch = stock.get_market_fundamental_by_ticker(self.today.strftime("%Y%m%d"), market="ALL")
        fetch['PER'] = fetch['PER'].apply(lambda val: val if val < 2000 else 0)
        self.base = self.base.join(
            other=fetch[["PER", "PBR", "DIV"]],
            how='left'
        )
        self.base.sort_values(by=['시가총액'], inplace=True, ascending=False)
        return

    def save(self) -> None:
        """
        마켓 데이터 저장
        :return:
        """
        self.base.reset_index(level=0, inplace=True)
        self.base.to_csv(os.path.join(warehouse, 'market/market.csv'), encoding='utf-8', index=False)
        self.base.to_csv(
            os.path.join(warehouse, f'market/{self.today.strftime("%Y%m%d")}market.csv'),
            encoding='utf-8', index=False
        )
        return


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)

    app = market(
        # date=datetime(2021, 10, 13)
    )
    app.update_percentage()
    app.update_multiple()
    app.save()