import os, random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter


__root__ = os.path.dirname(os.path.dirname(__file__))
__meta__ = pd.DataFrame()
class toolkit:
    @staticmethod
    def __mta__() -> None:
        global __meta__
        __meta__ = pd.read_csv(
            'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/group/WI26.csv',
            encoding='utf-8',
            index_col='종목코드'
        ).join(
            other=pd.read_csv(
                'http://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/market/market.csv',
                encoding='utf-8',
                index_col='종목코드'
            ).drop(columns=['종목명']),
            how='left'
        )[['섹터', '종목명', '종가', '시가총액', 'R1D', 'R1W', 'R1M', 'PER', 'DIV']].rename(
            columns={'R1D': '1일등락', 'R1W': '1주등락', 'R1M': '1개월등락'}
        )
        for col in __meta__.columns:
            if '등락' in col:
                __meta__[col] = round(__meta__[col], 2).astype(str) + '%'
        __meta__['PER'] = round(__meta__['PER'], 2)
        __meta__['종가'] = __meta__['종가'].apply(lambda p: '{:,}원'.format(int(p)))
        cap = (__meta__["시가총액"] / 100000000).astype(int).astype(str)
        __meta__['시가총액'] = cap.apply(lambda v: v + "억" if len(v) < 5 else v[:-4] + '조 ' + v[-4:] + '억')
        __meta__.index = __meta__.index.astype(str).str.zfill(6)
        return

    @staticmethod
    def __sma__(base:pd.Series, windows:list) -> pd.DataFrame:
        """
        단순 이동 평균(Simple Moving Average) 필터
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :return:
        """
        return pd.concat(
            objs={f'SMA{win}D': base.rolling(window=win).mean() for win in windows},
            axis=1
        )

    @staticmethod
    def __ema__(base:pd.Series, windows:list) -> pd.DataFrame:
        """
        지수 이동 평균(Exponent Moving Average) 필터
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :return:
        """
        return pd.concat(
            objs={f'EMA{win}D':base.ewm(span=win).mean() for win in windows},
            axis=1
        )

    @staticmethod
    def __iir__(base:pd.Series, windows:list, order:int=1) -> pd.DataFrame:
        """
        scipy 패키지 Butterworth 기본 제공 필터 (IIR :: Uses Feedback)
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :param order: 필터 차수
        :return:
        """
        objs = {}
        for cutoff in windows:
            normal_cutoff = (252 / cutoff) / (252 / 2)
            coeff_a, coeff_b = butter(order, normal_cutoff, btype='low', analog=False)
            objs[f'IIR{cutoff}D'] = pd.Series(data=filtfilt(coeff_a, coeff_b, base), index=base.index)
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def __fir__(base:pd.Series, windows:list) -> pd.DataFrame:
        """
        DEPRECATED :: scipy 패키지 FIR 필터
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :return:
        """
        objs = {}
        for cutoff in windows:
            normal_cutoff = (252 / cutoff) / (252 / 2)
            '''
            ripple ::
            width :: 클수록 Delay 상쇄/필터 성능 저하
            '''
            ripple, width = {
                5:  (10, 75 / (252 / 2)),
                10: (12, 75 / (252 / 2)),
                20: (20, 75 / (252 / 2)),
                60: (60, 75 / (252 / 2)),
                120: (80, 75 / (252 / 2)),
            }[cutoff]
            N, beta = kaiserord(ripple=ripple, width=width)
            taps = firwin(N, normal_cutoff, window=('kaiser', beta))
            objs[f'FIR{cutoff}D'] = pd.Series(data=lfilter(taps, 1.0, base), index=base.index)
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def __ans__(data: pd.DataFrame, by: str = '종가', td: int = 20, yld: float = 5.0) -> pd.DataFrame:
        """
        거래일(td) 기준 수익률(yld) 만족 지점 표기 데이터프레임
        :param data: 가격 정보 [시가, 저가, 고가, 종가] 포함 데이터프레임
        :param by: 기준 가격 정보
        :param td: 목표 거래일
        :param yld: 목표 수익률
        :return:
        """
        calc = data[['시가', '저가', '고가', '종가']].copy()
        pass_fail = [False] * len(calc)
        for i in range(td, len(calc), 1):
            afters = calc[i + 1:i + td + 1].values.flatten()
            if afters[0] == 0:
                continue

            for after in afters[1:]:
                if after == 0:
                    continue
                if 100 * (after / afters[0] - 1) >= yld:
                    pass_fail[i] = True
                    break
        calc['달성여부'] = pass_fail

        scale = ['#F63538', '#BF4045', '#8B444E', '#414554', '#35764E', '#2F9E4F', '#30CC5A']
        thres = {
            5: [-3, -2, -1, 1, 2, 3],
            10: [-3, -2, -1, 1, 2, 3],
            15: [-4, -2.5, -1, 1, 2.5, 4],
            20: [-5, -3, -1, 1, 3, 5],
        }
        for day, bound in thres.items():
            calc[f'{day}TD수익률'] = round(100 * calc[by].pct_change(periods=day).shift(-day).fillna(0), 2)
            cindex = [calc[f'{day}TD수익률'].min()] + bound + [calc[f'{day}TD수익률'].max()]
            calc[f'{day}TD색상'] = pd.cut(calc[f'{day}TD수익률'], bins=cindex, labels=scale, right=True)
            calc[f'{day}TD색상'].fillna(scale[0])
        return calc.drop(columns=['시가', '저가', '고가', '종가'])


class asset(toolkit):
    def __init__(self, **kwargs):
        """
        :param ticker: 종목코드
        :param kwargs:
        | KEYWORDS | TYPE |               COMMENT |                         POSSIBLE INPUTS |              DEFAULT |
        |     meta | pd.D |       외부 메타데이터 | columns=[종목코드, 종목명] 데이터프레임 |                *필수 |
        |   ticker |  str |              종목코드 |                                       - |               005930 |
        |      src |  str |      가격 데이터 출처 |                       online or offline |               online |
        |  windows | list | 필터 적용 거래일 모음 |                    [int, int, ..., int] | [5, 10, 20, 60, 120] |
        | filterby |  str |   필터 적용 기준 가격 |            시가 or 고가 or 저가 or 종가 |                 저가 |
        """
        global __meta__
        if __meta__.empty:
            self.__mta__()

        keys = kwargs.keys()
        self.ticker = kwargs['ticker'] if 'ticker' in keys else '005930'
        self.src = kwargs['src'] if 'src' in keys else 'online'
        self.windows = kwargs['windows'] if 'windows' in keys else [5, 10, 20, 60, 120]
        self.filterby = kwargs['filterby'] if 'filterby' in keys else '저가'


        self.name = __meta__.loc[self.ticker, '종목명']
        self.sector = __meta__.loc[self.ticker, '섹터']
        self.price = pd.read_csv(
            f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{self.ticker}.csv',
            encoding='utf-8',
            index_col='날짜'
        ) if self.src == 'online' else pd.read_csv(
            os.path.join(__root__, f'warehouse/series/{self.ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        self.price.index = pd.to_datetime(self.price.index)

        self._guide_ = pd.DataFrame()
        self._trend_ = pd.DataFrame()
        return

    @property
    def guide(self) -> pd.DataFrame:
        """
        주가 가이던스 (필터 선)
        :return:
        """
        if not self._guide_.empty:
            return self._guide_
        self._guide_ = pd.concat(objs=[
            self.__sma__(base=self.price[self.filterby], windows=self.windows),
            self.__ema__(base=self.price[self.filterby], windows=self.windows + [12, 26]),
            self.__iir__(base=self.price[self.filterby], windows=self.windows, order=1),
            self.__fir__(base=self.price[self.filterby], windows=self.windows)
        ], axis=1)
        return self._guide_

    @property
    def trend(self) -> pd.DataFrame:
        """
        주가 추세선
        :return:
        """
        if not self._trend_.empty:
            return self._trend_
        dat = self.guide.copy()
        self._trend_ = pd.concat(objs={
            '중장기IIR': dat['IIR60D'] - dat['EMA120D'],
            '중기IIR': dat['IIR60D'] - dat['EMA60D'],
            '중단기IIR': dat['IIR20D'] - dat['EMA60D'],
            '중장기FIR': dat['FIR60D'] - dat['EMA120D'],
            '중기FIR': dat['FIR60D'] - dat['EMA60D'],
            '중단기FIR': dat['FIR20D'] - dat['EMA60D'],
            '중장기SMA': dat['SMA60D'] - dat['SMA120D'],
            '중단기SMA': dat['SMA20D'] - dat['SMA60D'],
            '중장기EMA': dat['EMA60D'] - dat['EMA120D'],
            '중단기EMA': dat['EMA20D'] - dat['EMA60D'],
            'MACD': dat['EMA12D'] - dat['EMA26D'],
        }, axis=1)
        for col in self._trend_.columns:
            self._trend_[f'd{col}'] = self._trend_[col].diff()
            self._trend_[f'd2{col}'] = self._trend_[col].diff().diff()
        return self._trend_


class datum(asset):
    def oscillation(self, date:datetime=None, filter_type:str='iir', gap:int=60) -> pd.DataFrame:
        """
        필터의 입력 날짜 안정화 기간 데이터프레임 :: IIR 필터 전용
        :param date: 입력 날짜
        :param filter_type: 필터 종류
        :param gap: 필터 안정화 추정 기간(일수)
        :return:
                   IIR5D        IIR10D  ...       IIR120D  2016-08-29
        0   35299.991673  35287.675816  ...  33670.341345           0
        1   35475.931962  35460.695978  ...  33816.541437           1
        2   35477.689747  35493.334151  ...  33950.437559           2
        ...          ...           ...                ...         ...
        57  35476.114012  35505.357966  ...  35885.212421          57
        58  35476.114012  35505.357966  ...  35913.651424          58
        59  35476.114012  35505.357966  ...  35910.011256          59
        """
        app = {'sma':self.__sma__, 'ema':self.__ema__, 'iir': self.__iir__, 'fir':self.__fir__}[filter_type.lower()]
        base = self.price[self.filterby].copy()

        time_span = base.index.tolist()
        date = date if date else random.sample(time_span, 1)[0]
        n_date = time_span.index(date)
        objs = []
        for n, d in enumerate(time_span[n_date : n_date + gap]):
            rebase = base[base.index <= d].copy()
            out = app(base=rebase, windows=[5, 10, 20, 60, 120])
            data = out.iloc[-(n+1)].to_dict()
            data[f'{date.strftime("%Y-%m-%d")}'] = n
            objs.append(data)
        return pd.DataFrame(data=objs)

    def answers(self, count:int=-1):
        """
        필터 안정화 추정 학습용 데이터 샘플 (정답지)
        :param count:
        :return:
        """
        base = self.price[self.filterby].copy()

        time_span = base.index.tolist()
        samples = time_span[121:-120] if count == -1 else random.sample(time_span[121:-120], count)
        objs = []
        for sample in samples:
            raw = self.oscillation(date=sample, filter_type='iir', gap=60)
            raw.drop(columns=[sample.strftime("%Y-%m-%d")], inplace=True)
            err = (100 * (raw.iloc[-1] / raw.iloc[0] - 1)).to_dict()
            objs.append(err)
        return pd.DataFrame(data=objs, index=samples)


class chart(asset):
    def layout(self, title:str='', xtitle:str='날짜', ytitle:str='') -> go.Layout:
        """
        기본 차트 레이아웃
        :param title: 차트 제목 
        :param xtitle: x축 이름
        :param ytitle: y축 이름
        :param y2title: 제2 y축 이름
        :return: 
        """
        return go.Layout(
            title=f'<b>{self.name}[{self.ticker}]</b> : {title}',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            legend=dict(traceorder='reversed'),
            yaxis=dict(
                title=f'{ytitle}',
                showgrid=True, gridcolor='lightgrey',
                zeroline=False,
                showticklabels=True, autorange=True,
            ),
            xaxis=dict(
                title=f'{xtitle}',
                showgrid=True, gridcolor='lightgrey',
                zeroline=False,
                showticklabels=True, autorange=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            ),
        )

    def guidance(self, show:bool=False) -> go.Figure:
        """
        주가 가이던스(필터 선) 차트
        :param show: True::즉시 Plot
        :return:
        """
        fig = go.Figure(layout=self.layout(title='주가 가이던스Guidance 차트', ytitle='가격(KRW)'))
        frm = self.guide.copy()
        frm = frm.join(self.price.drop(columns=['거래량']), how='left')
        for col in frm.columns:
            cond = col[-3:] == '60D' or col == self.filterby
            unit = '[-]' if col.endswith('D') else '원'
            fig.add_trace(go.Scatter(
                x=frm.index,
                y=frm[col],
                name=col,
                visible=True if cond else 'legendonly',
                meta=['{}/{}/{}'.format(d.year, d.month, d.day) for d in frm.index],
                hovertemplate=col + '<br>날짜: %{meta}<br>필터: %{y:,.2f}' + unit + '<extra></extra>'
            ))
        fig.add_trace(
            go.Candlestick(
                x=self.price.index,
                customdata=['{}/{}/{}'.format(d.year, d.month, d.day) for d in self.price.index],
                open=self.price['시가'],
                high=self.price['고가'],
                low=self.price['저가'],
                close=self.price['종가'],
                increasing_line=dict(color='red'),
                decreasing_line=dict(color='blue'),
                name='일봉',
                visible='legendonly',
                showlegend=True,
            )
        )
        if show:
            fig.show()
        return fig

    def trendy(self, show:bool=False) -> go.Figure:
        """
        주가 추세선 차트
        :param show: True::즉시 Plot
        :return:
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.layout = self.layout(
            title='주가 추세Trend 차트',
            ytitle='추세값[-]',
        )
        frm = self.trend.copy()

        dform = ['{}/{}/{}'.format(d.year, d.month, d.day) for d in frm.index]
        for col in frm.columns:
            if col.startswith('d'):
                continue
            fig.add_trace(
                go.Scatter(
                    x=frm.index,
                    y=frm[col],
                    customdata=dform,
                    name=col,
                    mode='lines',
                    showlegend=True,
                    visible=True if col.startswith('중장') else 'legendonly',
                    hovertemplate=col + '<br>추세:%{y:.3f}<br>날짜:%{customdata}<br><extra></extra>',
                ),
                secondary_y=False
            )
        fig.add_trace(
            go.Scatter(
                x=self.price.index,
                y=self.price[self.filterby],
                meta=['{}/{}/{}'.format(d.year, d.month, d.day) for d in self.price.index],
                name='종가',
                mode='lines',
                showlegend=True,
                visible=True,
                hovertemplate='날짜:%{meta}<br>' + self.filterby + ':%{y:,}원<extra></extra>',
            ),
            secondary_y=True
        )
        fig.update_layout(yaxis=dict(zeroline=True, zerolinecolor='grey', zerolinewidth=1))
        if show:
            fig.show()
        return fig
    
    def momentum(self, show:bool=False) -> go.Figure:
        """
        추세선 모멘텀 차트
        :param show: True::즉시 Plot
        :return: 
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        frm = self.trend[60:].copy()
        cols = [col for col in frm.columns if col.startswith('d')]
        for col in cols:
            cond = (col.endswith('FIR') or col.endswith('IIR')) and '중기' in col
            fig.add_trace(
                go.Scatter(
                    x=frm.index,
                    y=frm[col],
                    name=col,
                    meta=['{}/{}/{}'.format(d.year, d.month, d.day) for d in frm.index],
                    visible=True if cond else 'legendonly',
                    hovertemplate=col + '<br>날짜: %{meta}<br>값:%{y:.4f}<extra></extra>'
                ),
                row=2 if col.startswith('d2') else 1, col=1
            )
        fig.update_layout(self.layout(
            title='주가 모멘텀Momentum 차트',
            ytitle='정규값',
            xtitle='날짜'
        ))
        fig.update_yaxes(
            title='정규값',
            showgrid=True, gridcolor='lightgrey',
            zeroline=True, zerolinecolor='grey',
            showticklabels=True, autorange=True,
            row=2, col=1
        )
        fig.update_xaxes(
            showgrid=True, gridcolor='lightgrey',
            row=2, col=1
        )
        if show:
            fig.show()
        return fig

    def scatter(self, col:str, td:int=20, perf:float=5.0, show:bool=False) -> go.Figure:
        """
        지표(추세, 변화량, 모멘텀, 합성 등) 대비 수익률 산포도
        :param col: 지표 (데이터프레임 열 이름)
        :param td: 목표 거래일(달성 기간)
        :param perf: 목표 수익률
        :param show: True::즉시 Plot
        :return:
        """
        fig = go.Figure()
        fig.update_layout(self.layout(
            title=f'{col}-산포도'
        ))
        if show:
            fig.show()
        return fig


if __name__ == "__main__":

    charter = chart(ticker='000660')
    print(f"{charter.name}({charter.ticker})")
    # charter.guidance(show=True)
    # charter.trendy(show=True)
    charter.momentum(show=True)

    # dater = datum(ticker='000660')
    # print(f"{dater.name}({dater.ticker})")
    # test_frame = dater.oscillation(date=None)
    # test_sample = dater.sampler(count=-1)
    # print(test_sample)

    # verify = myChart.verifier(myChart.price['저가'], pick='btr', sample=100)
    # print(verify)
    # print(verify.describe())
    # myChart.guidance_stability(pick='lpf', show=True, sample=datetime(2014, 5, 30))

