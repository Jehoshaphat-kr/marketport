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
    dx = 1  # 1 day interval
    d_dx = FinDiff(0, dx, 1, acc=accuracy)  # acc=3 #for 5-point stencil, currenly uses +/-1 day only
    d2_dx2 = FinDiff(0, dx, 2, acc=accuracy)  # acc=3 #for 5-point stencil, currenly uses +/-1 day only

    def get_minmax(h):
        clarr = np.asarray(h, dtype=np.float64)
        mom, momacc = d_dx(clarr), d2_dx2(clarr)

        def numdiff_extrema(func):
            return [x for x in range(len(mom))
                    if func(x) and
                    (mom[
                         x] == 0 or  # either slope is 0, or it crosses from positive to negative with the closer to 0 of the two chosen or prior if a tie
                     (x != len(mom) - 1 and (
                             mom[x] > 0 > mom[x + 1] and h[x] >= h[x + 1] or  # mom[x] >= -mom[x+1]
                             mom[x] < 0 < mom[x + 1] and h[x] <= h[x + 1]) or  # -mom[x] >= mom[x+1]) or
                      x != 0 and (mom[x - 1] > 0 > mom[x] and h[x - 1] < h[x] or  # mom[x-1] < -mom[x] or
                                  mom[x - 1] < 0 < mom[x] and h[x - 1] > h[x])))]  # -mom[x-1] < mom[x])))]
        return lambda x: momacc[x] > 0, lambda x: momacc[x] < 0, numdiff_extrema

    minFunc, maxFunc, numdiff_extrema = get_minmax(h)
    return numdiff_extrema(minFunc), numdiff_extrema(maxFunc)


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


if __name__ == "__main__":
    api = prices(ticker='007070', src='local')
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
    print(api.trend)
    # print(api.stc)
    # print(api.vortex)
