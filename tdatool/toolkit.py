import tdatool as tt
import pandas as pd
import os
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter


class liner:

    @staticmethod
    def fetch(ticker:str, src:str) -> pd.DataFrame:
        """
        시가, 저가, 고가, 종가, 거래량 시계열 데이터프레임
        :param ticker: 종목코드
        :param src: 주가 데이터 소스
        :return:
        """
        __price__ = pd.read_csv(
            f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{ticker}.csv',
            encoding='utf-8',
            index_col='날짜'
        ) if src == 'online' else pd.read_csv(
            os.path.join(tt.root, f'warehouse/series/{ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        __price__.index = pd.to_datetime(__price__.index)
        return __price__

    @staticmethod
    def calc_guide(series:pd.Series, window:list) -> pd.DataFrame:
        """
        주가 가이드(필터) 데이터프레임
        :param series: 필터 기준 주가(시가, 고가, 저가, 종가)
        :param window: 필터 대상 거래일
        :return:
        """
        # FIR: SMA
        objs = {f'SMA{win}D': series.rolling(window=win).mean() for win in window}

        # FIR: EMA
        objs.update({f'EMA{win}D': series.ewm(span=win).mean() for win in window})
        for win in window:
            # IIR: BUTTERWORTH
            cutoff = (252 / win) / (252 / 2)
            coeff_a, coeff_b = butter(N=1, Wn=cutoff, btype='lowpass', analog=False, output='ba')
            objs[f'IIR{win}D'] = pd.Series(data=filtfilt(coeff_a, coeff_b, series), index=series.index)

            # FIR: KAISER
            N, beta = kaiserord(ripple={5:10, 10:12, 20:20, 60:60, 120:80}[win], width=75 / (252 / 2))
            taps = firwin(N, cutoff, window=('kaiser', beta))
            objs[f'FIR{win}D'] = pd.Series(data=lfilter(taps, 1.0, series), index=series.index)
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def calc_trend(dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        주가 필터 기반 추세 데이터프레임
        :param dataframe: 주가 가이드 데이터프레임
        :return:
        """
        combination = [
            ['중장기IIR', 'IIR60D', 'EMA120D'], ['중기IIR', 'IIR60D', 'EMA60D'], ['중단기IIR', 'IIR20D', 'EMA60D'],
            ['중장기FIR', 'FIR60D', 'EMA120D'], ['중기FIR', 'FIR60D', 'EMA60D'], ['중단기FIR', 'FIR20D', 'EMA60D'],
            ['중장기SMA', 'SMA60D', 'SMA120D'], ['중단기SMA', 'SMA20D', 'SMA60D'],
            ['중장기EMA', 'EMA60D', 'EMA120D'], ['중단기EMA', 'EMA20D', 'EMA60D']
        ]
        objs = {}
        for label, numerator, denominator in combination:
            basis = dataframe[numerator] - dataframe[denominator]
            objs[label] = basis
            objs[f'd{label}'] = basis.diff()
            objs[f'd2{label}'] = basis.diff().diff()
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def calc_macd(series:pd.Series) -> pd.DataFrame:
        """
        MACD 데이터프레임
        :param series:
        :return:
        """
        macd = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return pd.concat(objs={'MACD': macd, 'MACD-Sig': signal, 'MACD-Hist': hist}, axis=1)

    @staticmethod
    def detect(dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        주요 지표 변곡점 판단
        :param dataframe: 주요 지표
        :return:
        """
        objs = {}
        cols = [col for col in dataframe if not col.startswith('d') and not 'Hist' in col and not 'Sig' in col]
        for col in cols:
            is_macd = True if col.startswith('MACD') else False
            data = []
            tr = dataframe['MACD' if is_macd else col].values
            sr = dataframe['MACD-Sig' if is_macd else f'd{col}'].values
            for n, date in enumerate(dataframe.index[1:]):
                if (is_macd and tr[n-1] < sr[n-1] and tr[n] > sr[n]) or (not is_macd and sr[n-1] < 0 < sr[n]):
                    data.append([date, tr[n], 'Buy', 'triangle-up', 'red'])
                elif (is_macd and tr[n-1] > sr[n-1] and tr[n] < sr[n]) or (not is_macd and sr[n-1] > 0 > sr[n]):
                    data.append([date, tr[n], 'Sell', 'triangle-down', 'blue'])
                elif not is_macd and tr[n - 1] < 0 < tr[n]:
                    data.append([date, tr[n], 'Golden-Cross', 'star', 'gold'])
                elif not is_macd and tr[n - 1] > 0 > tr[n]:
                    data.append([date, tr[n], 'Dead-Cross', 'x', 'black'])
            objs[f'det{col}'] = pd.DataFrame(data=data, columns=['날짜', 'value', 'bs', 'symbol', 'color']).set_index(keys='날짜')
        return pd.concat(objs=objs, axis=1)