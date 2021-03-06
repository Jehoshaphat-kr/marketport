import math, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from ta import add_all_ta_features as lib
from tdatool.frame import finances
from findiff import FinDiff
np.seterr(divide='ignore', invalid='ignore')


def get_extrema(h, accuracy=8):
    """
    Customized Pivot Detection
    Originally from PyPI: trendln @https://github.com/GregoryMorse/trendln
    """
    dx = 1
    d_dx, d2_dx2 = FinDiff(0, dx, 1, acc=accuracy), FinDiff(0, dx, 2, acc=accuracy)
    def get_peak(h):
        arr = np.asarray(h, dtype=np.float64)
        mom, momacc = d_dx(arr), d2_dx2(arr)

        def _diff_extrema_(func):
            return [
                x for x in range(len(mom)) if func(x) and (
                    mom[x] == 0 or
                    (
                        x != len(mom) - 1 and (
                            mom[x] > 0 > mom[x + 1] and h[x] >= h[x + 1] or mom[x] < 0 < mom[x + 1] and h[x] <= h[x + 1]
                        ) or x != 0 and (
                            mom[x - 1] > 0 > mom[x] and h[x - 1] < h[x] or mom[x - 1] < 0 < mom[x] and h[x - 1] > h[x]
                        )
                    )
                )
            ]
        return lambda x: momacc[x] > 0, lambda x: momacc[x] < 0, _diff_extrema_

    minFunc, maxFunc, diff_extrema = get_peak(h)
    return diff_extrema(minFunc), diff_extrema(maxFunc)


class prices(finances):

    def __init__(self, ticker: str = '005930', src: str = 'github', period: int = 5, meta = None):
        super().__init__(ticker=ticker, meta=meta)

        # Fetch Price
        today = datetime.today()
        start = today - timedelta((365 * period) + 180)
        if src.lower() == 'local':
            df = pd.read_csv(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f'warehouse/series/{ticker}.csv'
                ), encoding='utf-8', index_col='날짜'
            )
            df.index = pd.to_datetime(df.index)
            self.price = df[df.index >= start]
        elif src.lower().endswith('krx'):
            from pykrx import stock
            self.price = stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"), ticker)
        elif src.lower().endswith('github'):
            master = 'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series'
            df = pd.read_csv(f'{master}/{ticker}.csv', encoding='utf-8', index_col='날짜')
            df.index = pd.to_datetime(df.index)
            self.price = df[df.index >= start]
        else:
            raise KeyError(f'Key Error: Not Possible argument for src = {src}. Must be local/pykrx/github')
        self.price_ori = self.price.copy()
        self.__fillz__()

        # Technical Analysis
        # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#trend-indicators
        self.p_lib = lib(self.price.copy(), open='시가', close='종가', low='저가', high='고가', volume='거래량')

        # Empty Property
        self._filters_ = pd.DataFrame()
        self._guidance_ = pd.DataFrame()
        self._bend_point_ = pd.DataFrame()
        self._pivot_ = pd.DataFrame()
        self._trend_ = pd.DataFrame()
        return

    def __fillz__(self):
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
    def bollinger(self) -> pd.DataFrame:
        """
        볼린저(Bollinger) 밴드
        :return:
        """
        return pd.concat(
            objs={
                '상한선': self.p_lib.volatility_bbh,
                '하한선': self.p_lib.volatility_bbl,
                '기준선': self.p_lib.volatility_bbm,
                '밴드폭': self.p_lib.volatility_bbw,
                '신호': self.p_lib.volatility_bbp
            }, axis=1
        )

    @property
    def macd(self) -> pd.DataFrame:
        """
        MACD: Moving Average Convergence & Divergence 데이터프레임
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
    def stc(self) -> pd.Series:
        """
        STC: Schaff Trend Cycle 데이터프레임
        :return:
        """
        sr = self.p_lib.trend_stc
        sr.name = 'STC'
        return sr

    @property
    def cci(self) -> pd.Series:
        """
        CCI: Commodity Channel Index 데이터프레임
        :return:
        """
        sr = self.p_lib.trend_cci
        sr.name = 'CCI'
        return sr

    @property
    def trix(self) -> pd.Series:
        """
        TRIX: Triple exponential moving average
        :return:
        """
        sr = self.p_lib.trend_trix
        sr.name = 'TRIX'
        return sr

    @property
    def rsi(self) -> pd.Series:
        """
        RSI: Relative Strength Index 데이터프레임
        :return:
        """
        sr = self.p_lib.momentum_rsi
        sr.name = 'RSI'
        return sr 
        
    @property
    def stoch_rsi(self) -> pd.DataFrame:
        """
        Stochastic RSI 데이터프레임
        :return: 
        """
        return pd.concat(
            objs={'STOCH-RSI': self.p_lib.momentum_stoch, 'STOCH-RSI-Sig': self.p_lib.momentum_stoch_signal}, axis=1
        )

    @property
    def vortex(self) -> pd.DataFrame:
        """
        주가 Vortex 데이터프레임
        :return:
        """
        return pd.concat(
            objs={
                'VORTEX(+)': self.p_lib.trend_vortex_ind_pos,
                'VORTEX(-)': self.p_lib.trend_vortex_ind_neg,
                'VORTEX-Diff': self.p_lib.trend_vortex_ind_diff
            }, axis=1
        )

    @property
    def pivot(self) -> pd.DataFrame:
        """
        Pivot 지점 데이터프레임임
       :return:
        """
        if self._pivot_.empty:
            price = self.price[self.price.index >= (self.price.index[-1] - timedelta(365))].copy()
            span = price.index

            dump, upper = get_extrema(h=price['고가'])
            upper_index = [span[n] for n in upper]
            upper_pivot = price[span.isin(upper_index)]['고가']

            lower, dump = get_extrema(h=price['저가'])
            lower_index = [span[n] for n in lower]
            lower_pivot = price[span.isin(lower_index)]['저가']
            self._pivot_ = pd.concat(objs={'고점':upper_pivot, '저점':lower_pivot}, axis=1)
        return self._pivot_

    @property
    def trend(self) -> pd.DataFrame:
        """
        평균 추세선
        :return:
        """
        def norm(frm:pd.DataFrame, period:str, kind:str) -> pd.Series:
            """
            :param frm: 기간 별 slice 가격 및 정수 index N 포함 데이터프레임
            :param period: '1Y', '6M', '3M'
            :param kind: '지지선', '저항선'
            :return:
            """
            is_resist = True if kind == '저항선' else False
            key = '고가' if is_resist else '저가'
            base = frm[key]
            tip_index = base[base == (base.max() if is_resist else base.min())].index[-1]
            right = base[base.index > tip_index].drop_duplicates(keep='last').sort_values(ascending=not is_resist)
            left = base[base.index < tip_index].drop_duplicates(keep='first').sort_values(ascending=not is_resist)

            r_trend = pd.DataFrame()
            l_trend = pd.DataFrame()
            for n, sr in enumerate([right, left]):
                prev_len = len(frm)
                for index in sr.index:
                    sample = frm[frm.index.isin([index, tip_index])][['N', key]]
                    slope, intercept, r_value, p_value, std_err = linregress(sample['N'], sample[key])
                    regress = slope * frm['N'] + intercept

                    curr_len = len(frm[frm[key] >= regress]) if is_resist else len(frm[frm[key] <= regress])
                    if curr_len > prev_len: continue

                    if n: l_trend = slope * frm['N'] + intercept
                    else: r_trend = slope * frm['N'] + intercept

                    if curr_len <= 3: break
                    else: prev_len = curr_len
            if r_trend.empty:
                series = l_trend
            elif l_trend.empty:
                series = r_trend
            else:
                r_e = math.sqrt((r_trend - frm[key]).pow(2).sum())
                l_e = math.sqrt((l_trend - frm[key]).pow(2).sum())
                series = r_trend if r_e < l_e else l_trend
            series.name = f'{period}표준{kind}'
            return series

        def mean(frm_price:pd.DataFrame, frm_pivot:pd.DataFrame, period:str, kind:str) -> pd.Series:
            is_resist = True if kind == '저항선' else False
            key = '고점' if is_resist else '저점'
            y = frm_pivot[key].dropna()
            x = frm_price[frm_price.index.isin(y.index)]['N']
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            series = slope * frm_price['N'] + intercept
            series.name = f'{period}평균{kind}'
            return series

        if self._trend_.empty:
            objs = []
            gaps = [('1Y', 365), ('6M', 183), ('3M', 91), ('2M', 42)]
            for period, days in gaps:
                on = self.price.index[-1] - timedelta(days)
                frm_price = self.price[self.price.index >= on].copy()
                frm_price['N'] = np.arange(len(frm_price)) + 1
                frm_pivot = self.pivot[self.pivot.index >= on]
                objs.append(mean(frm_price=frm_price, frm_pivot=frm_pivot, period=period, kind='저항선'))
                objs.append(mean(frm_price=frm_price, frm_pivot=frm_pivot, period=period, kind='지지선'))
                objs.append(norm(frm=frm_price, period=period, kind='저항선'))
                objs.append(norm(frm=frm_price, period=period, kind='지지선'))
            self._trend_ = pd.concat(objs=objs, axis=1)
        return self._trend_

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


if __name__ == "__main__":
    api = prices(ticker='035720', src='pykrx')
    print(api.name)
    # print(api.price)
    # print(api.filters)
    # print(api.guidance)
    # print(api.macd)
    # print(api.bend_point)
    # print(api.bend_point['detMACD'].dropna())
    # print(api.h_sup_res)
    # print(api.bollinger)
    # print(api.pivot)
    # print(api.trend)
    # print(api.stc)
    # print(api.vortex)
    print(api.rsi)
    print(api.stoch_rsi)
    print(api.cci)