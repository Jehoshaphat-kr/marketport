import os
import pandas as pd
import numpy as np
import tdatool as tt
from datetime import datetime, timedelta
from scipy.stats import linregress
from pykrx import stock
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter


today = datetime.today()
__root__ = os.path.dirname(os.path.dirname(__file__))
class technical:

    def __init__(self, ticker: str = '005930', src: str = 'git', period: int = 5):

        # Parameter
        self.ticker = ticker
        self.name = tt.meta.loc[ticker, '종목명']
        self.period = period

        # Fetch Price
        self.price = {
            'krx': self.__fetch1__(),
            'git': self.__fetch2__(),
            'local': self.__fetch3__()
        }[src]

        # Default Properties :: Calculate Filter, Trend Line, MACD
        self.filters = self.__filtering__()
        self.guidance = self.__guide__()
        self.macd = self.__macd__()

        # Empty Property
        self._bend_point_ = pd.DataFrame()
        self._thres_ = pd.DataFrame()
        self._pivot_ = pd.DataFrame()
        self._trend_ = pd.DataFrame()
        return

    def __fetch1__(self) -> pd.DataFrame:
        """
        시가, 저가, 고가, 종가, 거래량 가격 데이터프레임
        :return:
        """
        from_date = (today - timedelta((365 * self.period) + 180)).strftime("%Y%m%d")
        to_date = today.strftime("%Y%m%d")
        return stock.get_market_ohlcv_by_date(fromdate=from_date, todate=to_date, ticker=self.ticker)

    def __fetch2__(self) -> pd.DataFrame:
        """
        시가, 저가, 고가, 종가, 거래량 가격 데이터프레임
        :return:
        """
        df = pd.read_csv(
            f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{self.ticker}.csv',
            encoding='utf-8',
            index_col='날짜'
        )
        df.index = pd.to_datetime(df.index)
        return df[df.index >= (today - timedelta((365 * self.period) + 180))]

    def __fetch3__(self) -> pd.DataFrame:
        """
        시가, 저가, 고가, 종가, 거래량 가격 데이터프레임
        :return:
        """
        df = pd.read_csv(
            os.path.join(__root__, f'warehouse/series/{self.ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        df.index = pd.to_datetime(df.index)
        return df[df.index >= (today - timedelta((365 * self.period) + 180))]

    def __filtering__(self) -> pd.DataFrame:
        """
        주가 가이드(필터) 데이터프레임
        :return:
        """
        series = self.price['종가']
        window = [5, 10, 20, 60, 120]
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
            N, beta = kaiserord(ripple={5: 10, 10: 12, 20: 20, 60: 60, 120: 80}[win], width=75 / (252 / 2))
            taps = firwin(N, cutoff, window=('kaiser', beta))
            objs[f'FIR{win}D'] = pd.Series(data=lfilter(taps, 1.0, series), index=series.index)
        return pd.concat(objs=objs, axis=1)

    def __guide__(self) -> pd.DataFrame:
        """
        주가 전망 지수 데이터프레임
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
            basis = self.filters[numerator] - self.filters[denominator]
            objs[label] = basis
            objs[f'd{label}'] = basis.diff()
            objs[f'd2{label}'] = basis.diff().diff()
        return pd.concat(objs=objs, axis=1)

    def __macd__(self) -> pd.DataFrame:
        """
        MACD 데이터프레임
        :return:
        """
        series = self.price['종가']
        main = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
        assist = main.ewm(span=9, adjust=False).mean()
        hist = main - assist
        return pd.concat(objs={'MACD': main, 'MACD-Sig': assist, 'MACD-Hist': hist}, axis=1)

    @property
    def bend_point(self) -> pd.DataFrame:
        """
        추세 분석 변곡점 감지
        :return:
        """
        if not self._bend_point_.empty:
            return self._bend_point_

        df = pd.concat(objs=[self.guidance, self.macd], axis=1)
        objs = {}
        cols = [col for col in df if not col.startswith('d') and not 'Hist' in col and not 'Sig' in col]
        for col in cols:
            is_macd = True if col.startswith('MACD') else False
            data = []
            tr = df['MACD' if is_macd else col].values[1:]
            sr = df['MACD-Sig' if is_macd else f'd{col}'].values[1:]
            for n, date in enumerate(df.index[1:]):
                if (is_macd and tr[n - 1] < sr[n - 1] and tr[n] > sr[n]) or (not is_macd and sr[n - 1] < 0 < sr[n]):
                    data.append([date, tr[n], 'Buy', 'triangle-up', 'red'])
                elif (is_macd and tr[n - 1] > sr[n - 1] and tr[n] < sr[n]) or (not is_macd and sr[n - 1] > 0 > sr[n]):
                    data.append([date, tr[n], 'Sell', 'triangle-down', 'blue'])
                elif not is_macd and tr[n - 1] < 0 < tr[n]:
                    data.append([date, tr[n], 'Golden-Cross', 'star', 'gold'])
                elif not is_macd and tr[n - 1] > 0 > tr[n]:
                    data.append([date, tr[n], 'Dead-Cross', 'x', 'black'])
            objs[f'det{col}'] = pd.DataFrame(data=data, columns=['날짜', 'value', 'bs', 'symbol', 'color']).set_index(
                keys='날짜')
        self._bend_point_ = pd.concat(objs=objs, axis=1)
        return self._bend_point_

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
    def thres(self) -> pd.DataFrame:
        """
        수평 지지선/저항선 데이터프레임
        :return:
        """
        if self._thres_.empty:
            frm = self.price[self.price.index >= (self.price.index[-1] - timedelta(180))].copy()
            low = frm['저가']
            high = frm['고가']
            spread = (high - low).mean()

            def is_support(i):
                return low[i] < low[i - 1] < low[i - 2] and low[i] < low[i + 1] < low[i + 2]

            def is_resistance(i):
                return high[i] > high[i - 1] > high[i - 2] and high[i] > high[i + 1] > high[i + 2]

            def is_far_from_level(l, lines):
                return np.sum([abs(l - x) < spread for x in lines]) == 0

            levels = []
            data = []
            for n, date in enumerate(frm.index[2: len(frm) - 2]):
                if is_support(n) and is_far_from_level(l=low[n], lines=levels):
                    sample = (n, low[n])
                    levels.append(sample)
                    data.append(list(sample) + list((date, f'지지선@{date.strftime("%Y%m%d")[2:]}')))
                elif is_resistance(n) and is_far_from_level(l=frm['고가'][n], lines=levels):
                    sample = (n, high[n])
                    levels.append(sample)
                    data.append(list(sample) + list((date, f'저항선@{date.strftime("%Y%m%d")[2:]}')))
            self._thres_= pd.DataFrame(data=data, columns=['ID', '가격', '날짜', '종류']).set_index(keys='날짜')
        return self._thres_

    @property
    def bollinger(self) -> pd.DataFrame:
        """
        볼린저(Bollinger) 밴드
        :return:
        """
        basis_prc = self.filters['SMA20D']
        basis_std = self.price['종가'].rolling(window=20).std()
        objs = {'상한선':basis_prc + (2 * basis_std), '하한선':basis_prc - (2 * basis_std) , '기준선': basis_prc}
        return pd.concat(objs=objs, axis=1).dropna()

    @property
    def trend(self) -> pd.DataFrame:
        """
        추세선
        :return:
        """
        if self._trend_.empty:
            price = self.price[self.price.index >= (self.price.index[-1] - timedelta(365))].copy()
            span = price.index

            lower, upper = tt.calc_support_resistance(h=price['고가'])
            upper_pivot, upper_range, upper_trend, upper_window = upper

            lower, upper = tt.calc_support_resistance(h=price['저가'])
            lower_pivot, lower_range, lower_trend, lower_window = lower

            # Pivot Points
            data = []
            for n, date in enumerate(span):
                if n in upper_pivot and n in lower_pivot:
                    data.append([date, price['저가'][n], price['고가'][n]])
                elif n in lower_pivot:
                    data.append([date, price['저가'][n], np.nan])
                elif n in upper_pivot:
                    data.append([date, np.nan, price['고가'][n]])
            pivot = pd.DataFrame(data=data, columns=['날짜', 'PV-지지', 'PV-저항']).set_index(keys='날짜')

            # Average Trend
            cols = ['날짜', 'Avg-지지선', 'Avg-저항선']
            data = [[price.index[0], lower_range[1], upper_range[1]],
                    [price.index[-1], lower_range[0] * (len(price)-1) + lower_range[1], upper_range[0] * (len(price)-1) + upper_range[1]]]
            trend_avg = pd.DataFrame(data=data, columns=cols).set_index(keys='날짜')

            # Trend
            min_h, max_h = min(min(price['저가']), min(price['고가'])), max(max(price['저가']), max(price['고가']))
            data = []
            for label, line in [('저항선', upper_trend), ('지지선', lower_trend)]:
                h = price['저가' if label == '지지선' else '고가']
                for n, t in enumerate(line[:3]):
                    point, factor = t
                    maxx = point[-1] + 1
                    while maxx < len(price) - 1:
                        ypred = factor[0] * maxx + factor[1]
                        if (h[maxx - 1] < ypred < h[maxx] or h[maxx] < ypred < h[maxx - 1] or
                            ypred > max_h + (max_h - min_h) * 0.1 or ypred < min_h - (max_h - min_h) * 0.1): break
                        maxx += 1
                    x_vals = np.array((point[0], maxx))
                    y_vals = factor[0] * x_vals + factor[1]
                    x_date = [span[n] for n in x_vals]
                    data.append(pd.Series(data=y_vals, index=x_date, name=f'{label}{n + 1}'))
            trend_line = pd.concat(objs=data, axis=1)

            self._trend_ = pd.concat([pivot, trend_avg, trend_line], axis=1)
        return self._trend_


if __name__ == "__main__":
    api = technical(ticker='005930', src='local')
    # print(api.price)
    # print(api.filters)
    # print(api.guidance)
    # print(api.macd)
    # print(api.bend_point)
    # print(api.bend_point['detMACD'].dropna())
    # print(api.h_sup_res)
    # print(api.bollinger)
    # print(api.pivot)
    print(api.trend)