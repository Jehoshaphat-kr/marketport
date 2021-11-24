import pandas as pd
import numpy as np
import tdatool as tt
from tdatool.toolkit import liner
from datetime import datetime, timedelta
from scipy.stats import linregress


class timeseries(liner):

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
        self.__guide__ = self.calc_guide(series=self.__price__[self.filter_key], window=self.filter_win)
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
    def guide(self) -> pd.DataFrame:
        """
        주가 필터 선 데이터프레임
        :return:
        """
        return self.__guide__

    @property
    def trend(self) -> pd.DataFrame:
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
    def detector(self) -> pd.DataFrame:
        """
        추세 분석 변곡점 감지
        :return:
        """
        if self.__detector__.empty:
            self.__detector__ = self.detect(dataframe=pd.concat(objs=[self.trend, self.macd], axis=1))
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
    def limit(self) -> pd.DataFrame:
        """
        지지선/저항선 표기
        :return:
        """
        def is_support(df, i):
            _ = df['저가']
            support = _[i] < _[i - 1] < _[i - 2] and _[i] < _[i + 1] < _[i + 2]
            return support

        def is_resistance(df, i):
            _ = df['고가']
            resistance = _[i] > _[i - 1] > _[i - 2] and _[i] > _[i + 1] > _[i + 2]
            return resistance

        def is_far_from_level(l, s, lines):
            return np.sum([abs(l - x) < s for x in lines]) == 0

        frm = self.price[self.price.index >= (self.price.index[-1] - timedelta(180))].copy()
        s_hat = np.mean(frm['고가'] - frm['저가'])

        levels = []
        index = []
        types = []
        s_cnt = 1
        r_cnt = 1
        for n, date in enumerate(frm.index[2: len(frm) - 2]):
            if is_support(frm, n):
                if is_far_from_level(l=frm['저가'][n], s=s_hat, lines=levels):
                    levels.append((n, frm['저가'][n]))
                    index.append(date)
                    types.append(f'지지선{s_cnt}')
                    s_cnt += 1
            elif is_resistance(frm, n):
                if is_far_from_level(l=frm['고가'][n], s=s_hat, lines=levels):
                    levels.append((n, frm['고가'][n]))
                    index.append(date)
                    types.append(f'저항선{r_cnt}')
                    r_cnt += 1
        _limit_ = pd.DataFrame(levels, columns=['N', '레벨'], index=index)
        _limit_['종류'] = types
        return _limit_


class finances:
    def __init__(self, ticker:str):
        self.ticker = ticker

        self.__y_state__ = pd.DataFrame()
        self.__q_state__ = pd.DataFrame()
        return

    @property
    def y_state(self) -> pd.DataFrame:
        """
        연간 연결 재무제표 데이터프레임
        :return:
        """
        if not self.__y_state__.empty:
            return self.__y_state__

        return pd.DataFrame()

    @property
    def q_state(self) -> pd.DataFrame:
        """
        분기 연결 재무제표 데이터프레임
        :return:
        """
        if not self.__q_state__.empty:
            return self.__q_state__
        return pd.DataFrame()

if __name__ == "__main__":
    api = timeseries(ticker='005930')
    # print(api.price)
    # print(api.guide)
    # print(api.trend)
    # print(api.macd)
    # print(api.detector)
    # print(api.detector['detMACD'].dropna())