import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter
from scipy.stats import linregress


class timeseries:
    series = pd.Series()
    filter_win = list()
    @property
    def price(self) -> pd.DataFrame:


    @property
    def sma(self) -> pd.DataFrame:
        """
        단순 이동 평균(Simple Moving Average) 필터
        :return:
        """
        return pd.concat(objs={f'SMA{win}D': self.series.rolling(window=win).mean() for win in self.filter_win}, axis=1)

    @property
    def ema(self) -> pd.DataFrame:
        """
        지수 이동 평균(Exponent Moving Average) 필터
        :return:
        """
        return pd.concat(objs={f'EMA{win}D': self.series.ewm(span=win).mean() for win in self.filter_win}, axis=1)

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
            objs[f'IIR{cutoff}D'] = pd.Series(data=filtfilt(coeff_a, coeff_b, self.series), index=self.series.index)
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
            objs[f'FIR{cutoff}D'] = pd.Series(data=lfilter(taps, 1.0, self.series), index=self.series.index)
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