import os
import pandas as pd
import numpy as np
import tdatool as tt
from tdatool.charts import chart
from datetime import datetime, timedelta
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter
from scipy.stats import linregress


class TimeSeries(chart):
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

        self.__price__ = pd.DataFrame()
        self.__guide__ = pd.DataFrame()
        return

    @property
    def price(self) -> pd.DataFrame:
        """
        시가, 저가, 고가, 종가, 거래량 시계열 데이터프레임
        :return:
        """
        if not self.__price__.empty:
            return self.__price__

        self.__price__ = pd.read_csv(
            f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{self.ticker}.csv',
            encoding='utf-8',
            index_col='날짜'
        ) if self.data_src == 'online' else pd.read_csv(
            os.path.join(tt.root, f'warehouse/series/{self.ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        self.__price__.index = pd.to_datetime(self.__price__.index)
        return self.__price__

    @property
    def guide(self) -> pd.DataFrame:
        """
        주가 필터 선 데이터프레임
        :return:
        """
        if not self.__guide__.empty:
            return self.__guide__
        self.__guide__ = pd.concat(objs=[self.sma, self.ema, self.fir, self.iir], axis=1)
        return self.__guide__

    @property
    def sma(self) -> pd.DataFrame:
        """
        단순 이동 평균(Simple Moving Average) 필터
        :return:
        """
        return pd.concat(
            objs={f'SMA{win}D': self.price[self.filter_key].rolling(window=win).mean() for win in self.filter_win},
            axis=1
        )

    @property
    def ema(self) -> pd.DataFrame:
        """
        지수 이동 평균(Exponent Moving Average) 필터
        :return:
        """
        return pd.concat(
            objs={f'EMA{win}D': self.price[self.filter_key].ewm(span=win).mean() for win in self.filter_win},
            axis=1
        )

    @property
    def iir(self) -> pd.DataFrame:
        """
        scipy 패키지 Butterworth 기본 제공 필터 (IIR :: Uses Feedback)
        :return:
        """
        objs = {}
        for cutoff in self.filter_win:
            normal_cutoff = (252 / cutoff) / (252 / 2)
            coeff_a, coeff_b = butter(N=1, Wn=normal_cutoff, btype='lowpass', analog=False, output='ba')
            objs[f'IIR{cutoff}D'] = pd.Series(
                data=filtfilt(coeff_a, coeff_b, self.price[self.filter_key]),
                index=self.price[self.filter_key].index
            )
        return pd.concat(objs=objs, axis=1)

    @property
    def fir(self) -> pd.DataFrame:
        """
        DEPRECATED :: scipy 패키지 FIR 필터
        :return:
        """
        objs = {}
        for cutoff in self.filter_win:
            normal_cutoff = (252 / cutoff) / (252 / 2)
            '''
            ripple ::
            width :: 클수록 Delay 상쇄/필터 성능 저하
            '''
            ripple, width = {
                5: (10, 75 / (252 / 2)),
                10: (12, 75 / (252 / 2)),
                20: (20, 75 / (252 / 2)),
                60: (60, 75 / (252 / 2)),
                120: (80, 75 / (252 / 2)),
            }[cutoff]
            N, beta = kaiserord(ripple=ripple, width=width)
            taps = firwin(N, normal_cutoff, window=('kaiser', beta))
            objs[f'FIR{cutoff}D'] = pd.Series(
                data=lfilter(taps, 1.0, self.price[self.filter_key]),
                index=self.price[self.filter_key].index
            )
        return pd.concat(objs=objs, axis=1)

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
            objs[f'{label}추세(상)'] = slope * df['X'] + intercept

            while len(df_dn) > 3:
                slope, intercept, r_value, p_value, std_err = linregress(x=df_dn['X'], y=df_dn['저가'])
                df_dn = df_dn[df_dn['저가'] <= (slope * df_dn['X'] + intercept)]
            slope, intercept, r_value, p_value, std_err = linregress(x=df_dn['X'], y=df_dn['저가'])
            objs[f'{label}추세(하)'] = slope * df['X'] + intercept
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

        levels = [];
        index = [];
        types = [];
        s_cnt = 1;
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
        self._limit_ = pd.DataFrame(levels, columns=['N', '레벨'], index=index)
        self._limit_['종류'] = types
        return self._limit_