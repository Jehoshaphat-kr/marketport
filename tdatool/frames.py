import pandas as pd
import numpy as np
import tdatool as tt
from tdatool.toolkit import technical, fundamental
from datetime import datetime, timedelta
from scipy.stats import linregress
from pykrx import stock


class timeseries(technical):

    def __init__(
        self,
        ticker: str = '005930',
        data_src: str = 'online',
        filter_key: str = '종가',
        filter_win: list = None,
        sr_diverse: bool = False
    ):
        self.name = tt.meta.loc[ticker, '종목명']
        self.ticker = ticker
        self.data_src = data_src
        self.filter_key = filter_key
        self.filter_win = [5, 10, 20, 60, 120] if not filter_win else filter_win
        self.sr_diverse = sr_diverse

        self.__price__ = self.fetch(ticker=self.ticker, src=self.data_src)
        self.__guide__ = self.calc_filter(series=self.__price__[self.filter_key], window=self.filter_win)
        self.__trend__ = self.calc_trend(self.__guide__)
        self.__macd__ = self.calc_macd(series=self.__price__[self.filter_key])
        self.__detector__ = pd.DataFrame()
        return

    @property
    def price(self) -> pd.DataFrame:
        """
        시가, 저가, 고가, 종가, 거래량 시계열 데이터프레임
        :return:
        """
        return self.__price__

    @property
    def filter_line(self) -> pd.DataFrame:
        """
        주가 필터 선 데이터프레임
        :return:
        """
        return self.__guide__

    @property
    def trend_line(self) -> pd.DataFrame:
        """
        일반 주가 추세 분석
        :return:
        """
        return self.__trend__

    @property
    def macd(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Moving Average Convergence and Divergence
        :return:
        """
        return self.__macd__

    @property
    def trade_points(self) -> pd.DataFrame:
        """
        추세 분석 변곡점 감지
        :return:
        """
        if self.__detector__.empty:
            self.__detector__ = self.calc_trade_points(dataframe=pd.concat(objs=[self.trend_line, self.macd], axis=1))
        return self.__detector__

    @property
    def bound(self):
        """
        기간별 주가 진동 범위
        :return:
        """
        objs = {}
        for dt, label in [(365, '1Y'), (0, 'YTD'), (182, '6M'), (91, '3M')]:
            df = self.price[
                self.price.index >= datetime(datetime.today().year, 1, 1)
            ].copy() if label == 'YTD' else self.price[
                self.price.index >= (datetime.today() - timedelta(dt))
            ].copy()

            df['X'] = np.arange(len(df)) + 1
            df_up = df.copy()
            df_dn = df.copy()
            while len(df_up) > 3:
                slope, intercept, r_value, p_value, std_err = linregress(x=df_up['X'], y=df_up['고가'])
                df_up = df_up[df_up['고가'] > (slope * df_up['X'] + intercept)]
            slope, intercept, r_value, p_value, std_err = linregress(x=df_up['X'], y=df_up['고가'])
            objs[f'{label}추세UP'] = slope * df['X'] + intercept

            while len(df_dn) > 3:
                slope, intercept, r_value, p_value, std_err = linregress(x=df_dn['X'], y=df_dn['저가'])
                df_dn = df_dn[df_dn['저가'] <= (slope * df_dn['X'] + intercept)]
            slope, intercept, r_value, p_value, std_err = linregress(x=df_dn['X'], y=df_dn['저가'])
            objs[f'{label}추세DN'] = slope * df['X'] + intercept
        return pd.concat(objs=objs, axis=1)

    @property
    def h_support_resistance(self) -> pd.DataFrame:
        """
        지지선/저항선 데이터프레임
        :return:
        """
        return self.calc_horizontal_line(dataframe=self.price)



class finances(fundamental):
    def __init__(self, ticker:str):
        self.ticker = ticker
        self.name = tt.meta.loc[ticker, '종목명']
        self.annual_statement, self.quarter_statement = self.fetch_statement(ticker=ticker)
        self.summary = self.fetch_summary(ticker=ticker)
        self.update_cap()
        return

    def update_cap(self) -> None:
        """
        기말 시가총액 정보 추가
        :return:
        """
        a_period = self.annual_statement.index
        fromdate = a_period[0].replace('/','') + '20'
        todate = datetime.today().strftime("%Y%m%d")
        cap = stock.get_market_cap_by_date(fromdate=fromdate, todate=todate, ticker=self.ticker, freq='m')
        cap['ID'] = [date.strftime("%Y/%m") for date in cap.index]
        cap['시가총액'] = (cap['시가총액']/100000000).astype(int)

        a_key = [i[:-3] if i.endswith(')') else i for i in self.annual_statement.index]
        q_key = [i[:-3] if i.endswith(')') else i for i in self.quarter_statement.index]
        a_cap = cap[cap['ID'].isin(a_key)][['ID', '시가총액']].copy().set_index(keys='ID')
        q_cap = cap[cap['ID'].isin(q_key)][['ID', '시가총액']].copy().set_index(keys='ID')
        self.annual_statement = self.annual_statement.join(a_cap, how='left')
        self.quarter_statement = self.quarter_statement.join(q_cap, how='left')
        return


if __name__ == "__main__":
    # api = timeseries(ticker='005930')
    # print(api.price)
    # print(api.guide)
    # print(api.trend)
    # print(api.macd)
    # print(api.detector)
    # print(api.detector['detMACD'].dropna())

    api = finances(ticker='000660')
    # print(api.annual_statement)
    # print(api.quarter_statement)
