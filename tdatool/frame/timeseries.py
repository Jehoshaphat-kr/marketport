import pandas as pd
import numpy as np
import tdatool.frame as frm
from datetime import timedelta
from pykrx import stock
from scipy.signal import butter, filtfilt
from ta import add_all_ta_features as lib

class analytic:

    def __init__(self, ticker: str = '005930', src: str = 'git', period: int = 5):

        # Parameter
        self.ticker = ticker
        self.name = stock.get_market_ticker_name(ticker=ticker)

        # Fetch Price
        self.price = frm.fetch(ticker=ticker, src=src, period=period)
        self.fillz()
        self.p_lib = lib(self.price, open='시가', close='종가', low='저가', high='고가', volume='거래량')
        # print(self.p_lib)
        print(self.p_lib.columns)

        # Empty Property
        self._filters_ = pd.DataFrame()
        self._guidance_ = pd.DataFrame()
        self._macd_ = pd.DataFrame()
        self._bend_point_ = pd.DataFrame()
        self._thres_ = pd.DataFrame()
        self._pivot_ = pd.DataFrame()
        self._trend_ = pd.DataFrame()
        return

    def fillz(self):
        """
        거래 중지 이력 Fill Data 전처리
        :return:
        """
        if 0 in self.price['시가'].tolist():
            p = self.price.copy()
            data = []
            for o, c, l, h, v in zip(p.시가, p.종가, p.저가, p.고가, p.거래량):
                if o == 0:
                    data.append([c, c, c, c, v])
                else:
                    data.append([o, c, l, h, v])
            self.price = pd.DataFrame(data=data, columns=['시가', '종가', '저가', '고가', '거래량'], index=p.index)
        return

    @property
    def filters(self) -> pd.DataFrame:
        """
        주가 가이드(필터) 데이터프레임
        :return:
        """
        if self._filters_.empty:
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
            self._filters_ = pd.concat(objs=objs, axis=1)
        return self._filters_

    @property
    def guidance(self) -> pd.DataFrame:
        """
        주가 전망 지수 데이터프레임
        :return:
        """
        if self._guidance_.empty:
            combination = [
                ['중장기IIR', 'IIR60D', 'EMA120D'], ['중기IIR', 'IIR60D', 'EMA60D'], ['중단기IIR', 'IIR20D', 'EMA60D'],
                ['중장기SMA', 'SMA60D', 'SMA120D'], ['중단기SMA', 'SMA20D', 'SMA60D'],
                ['중장기EMA', 'EMA60D', 'EMA120D'], ['중단기EMA', 'EMA20D', 'EMA60D']
            ]
            objs = {}
            for label, numerator, denominator in combination:
                basis = self.filters[numerator] - self.filters[denominator]
                objs[label] = basis
                objs[f'd{label}'] = basis.diff()
                objs[f'd2{label}'] = basis.diff().diff()
            self._guidance_ = pd.concat(objs=objs, axis=1)
        return self._guidance_

    @property
    def bollinger(self) -> pd.DataFrame:
        """
        볼린저(Bollinger) 밴드
        :return:
        """
        hi = self.p_lib.volatility_bbhi * self.p_lib.volatility_bbh
        li = self.p_lib.volatility_bbli * self.p_lib.volatility_bbl
        return pd.concat(
            objs={
                '상한선': self.p_lib.volatility_bbh,
                '하한선': self.p_lib.volatility_bbl,
                '기준선': self.p_lib.volatility_bbm,
                '상한지시': hi.drop(index=hi[hi == 0].index),
                '하한지시': li.drop(index=li[li == 0].index),
                '밴드폭': self.p_lib.volatility_bbw,
                '신호': self.p_lib.volatility_bbp
            }, axis=1
        ).dropna()

    @property
    def macd(self) -> pd.DataFrame:
        """
        MACD 데이터프레임
        :return:
        """
        return pd.concat(
            objs={
                'MACD':self.p_lib.trend_macd,
                'MACD-Sig':self.p_lib.trend_macd_signal,
                'MACD-Hist':self.p_lib.trend_macd_diff
            }, axis=1
        )

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
    def trend(self) -> pd.DataFrame:
        """
        추세선
        :return:
        """
        if self._trend_.empty:
            price = self.price[self.price.index >= (self.price.index[-1] - timedelta(365))].copy()
            span = price.index

            lower, upper = frm.calc_support_resistance(h=price['고가'])
            upper_pivot, upper_range, upper_trend, upper_window = upper

            lower, upper = frm.calc_support_resistance(h=price['저가'])
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
    api = analytic(ticker='005930', src='local')
    # print(api.price)
    # print(api.filters)
    # print(api.guidance)
    # print(api.macd)
    # print(api.bend_point)
    # print(api.bend_point['detMACD'].dropna())
    # print(api.h_sup_res)
    print(api.bollinger)
    # print(api.pivot)
    # print(api.trend)