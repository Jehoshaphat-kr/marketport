import os, random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as of
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter
from scipy.stats import linregress


__root__ = os.path.dirname(os.path.dirname(__file__))
__meta__ = pd.DataFrame()
class toolkit:
    @staticmethod
    def __mta__() -> None:
        global __meta__
        __meta__ = pd.read_csv(
            'http://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/market/market.csv',
            encoding='utf-8',
            index_col='종목코드'
        ).join(
            other=pd.read_csv(
                'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/group/WI26.csv',
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
            coeff_a, coeff_b = butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False, output='ba')
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
    def __ans__(data: pd.DataFrame, by: str = '종가') -> pd.DataFrame:
        """
        거래일(td) 기준 수익률(yld) 만족 지점 표기 데이터프레임
        :param data: 가격 정보 [시가, 저가, 고가, 종가] 포함 데이터프레임
        :param by: 기준 가격 정보
        :return:
        """
        calc = data[['시가', '저가', '고가', '종가']].copy()
        pass_fail = [False] * len(calc)
        for td, yld in [(10, 3.0), (15, 4.0), (20, 5.0), (25, 6.0)]:
            for i in range(len(calc)):
                afters = calc[(i + 1) : (i + td + 1)].values.flatten()
                if len(afters) < 4:
                    continue
                if afters[0] == 0:
                    continue

                for after in afters[1:]:
                    if after == 0:
                        continue
                    if 100 * (after / afters[0] - 1) >= yld:
                        pass_fail[i] = True
                        break
            calc[f'GET-{td}TD{int(yld)}P'] = pass_fail

        for td, bound in {
            10: [-3, -2, -1, 1, 2, 3],
            15: [-4, -2.5, -1, 1, 2.5, 4],
            20: [-5, -3, -1, 1, 3, 5],
            25: [-6, -4, -1.5, 1.5, 4, 6],
        }.items():
            calc[f'PERF-{td}TD'] = round(100 * calc[by].pct_change(periods=td).shift(-td).fillna(0), 2)
            calc[f'COLOR-{td}TD'] = pd.cut(
                calc[f'PERF-{td}TD'],
                bins=[calc[f'PERF-{td}TD'].min()] + bound + [calc[f'PERF-{td}TD'].max()],
                labels=['#F63538', '#BF4045', '#8B444E', '#414554', '#35764E', '#2F9E4F', '#30CC5A'],
                right=True
            )
            calc[f'COLOR-{td}TD'].fillna('#F63538')
        return calc.drop(columns=['시가', '저가', '고가', '종가'])

    def guides(self, base:pd.Series, windows:list):
        """
        일반 주가 가이던스 분석
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :return:
        """
        return pd.concat(objs=[
            self.__sma__(base=base, windows=windows),
            self.__ema__(base=base, windows=windows + [12, 26]),
            self.__iir__(base=base, windows=windows, order=1),
            self.__fir__(base=base, windows=windows)
        ], axis=1)

    def trends(self, base:pd.Series, windows:list):
        """
        일반 주가 추세 분석
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :return:
        """
        cumulate = 100 * ((base.pct_change().fillna(0) + 1).cumprod() - 1)
        rebase = self.guides(base=cumulate, windows=windows)
        frame = pd.concat(objs={
            '중장기IIR': rebase['IIR60D'] - rebase['EMA120D'],
            '중기IIR': rebase['IIR60D'] - rebase['EMA60D'],
            '중단기IIR': rebase['IIR20D'] - rebase['EMA60D'],
            '중장기FIR': rebase['FIR60D'] - rebase['EMA120D'],
            '중기FIR': rebase['FIR60D'] - rebase['EMA60D'],
            '중단기FIR': rebase['FIR20D'] - rebase['EMA60D'],
            '중장기SMA': rebase['SMA60D'] - rebase['SMA120D'],
            '중단기SMA': rebase['SMA20D'] - rebase['SMA60D'],
            '중장기EMA': rebase['EMA60D'] - rebase['EMA120D'],
            '중단기EMA': rebase['EMA20D'] - rebase['EMA60D'],
            'MACD': rebase['EMA12D'] - rebase['EMA26D'],
        }, axis=1)
        for col in frame.columns:
            frame[f'd{col}'] = frame[col].diff()
            frame[f'd2{col}'] = frame[col].diff().diff()
        return frame
    
class chart:
    def __init__(self, name:str, ticker:str):
        """
        :param name: 종목명 
        :param ticker: 종목코드
        """
        self.name = name
        self.ticker = ticker
        return
    
    @staticmethod
    def format(date_list: np.array) -> np.array:
        """
        날짜 형식 변경 (from)datetime --> (to)YY/MM/DD
        :param date_list: 날짜 리스트
        :return:
        """
        return [f'{d.year}/{d.month}/{d.day}' for d in date_list]

    def layout(self, title:str = '', xtitle:str='날짜', ytitle:str='') -> go.Layout:
        """
        기본 차트 레이아웃
        :param title: 차트 제목 
        :param xtitle: x축 이름
        :param ytitle: y축 이름
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
            xaxis_rangeslider=dict(visible=False)
        )

    def default(self, base:pd.DataFrame, support_resist:pd.DataFrame, show:bool=False, save:bool=False) -> go.Figure:
        """
        기본 주가 분석
        :param base:
        :param support_resist:
        :param show:
        :param save:
        :return:
        """
        data = base[base.index >= base.index[-1] - timedelta(5*365)].copy()
        fig = make_subplots(rows=2, cols=1, row_width=[0.15, 0.85], shared_xaxes=True, vertical_spacing=0.05)
        
        # 거래량
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['거래량'],
            customdata=self.format(date_list=data.index),
            name='거래량',
            marker=dict(
                color=['blue' if data.loc[d, '시가'] > data.loc[d, '종가'] else 'red' for d in data.index]
            ),
            showlegend=False,
            hovertemplate='날짜:%{customdata}<br>거래량:%{y:,}<extra></extra>'
        ), row=2, col=1)
        
        # 일봉
        fig.add_trace(go.Candlestick(
            x=data.index,
            customdata=self.format(date_list=data.index),
            open=data['시가'],
            high=data['고가'],
            low=data['저가'],
            close=data['종가'],
            increasing_line=dict(color='red'),
            decreasing_line=dict(color='blue'),
            name='일봉',
            visible='legendonly',
            showlegend=True,
        ), row=1, col=1)
        
        # 개별 가격
        for col in ['시가', '고가', '저가', '종가']:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                customdata=self.format(date_list=data.index),
                visible=True if col == '종가' else 'legendonly',
                hovertemplate=col + '<br>날짜:%{customdata}<br>가격:%{y:,}원<extra></extra>'
            ), row=1, col=1)

        # 추세선
        for col in data.columns:
            if not '추세' in col:
                continue
            gap = [l for l in ['1Y', 'YTD', '6M', '3M'] if col.startswith(l)][0]
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                legendgroup=f'{gap}추세선',
                name=f'{gap}추세선',
                visible='legendonly',
                showlegend=True if col.endswith('(상)') else False,
                hovertemplate=col + '<extra></extra>'
            ))

        # 지지/저항선
        for n, date in enumerate(support_resist.index):
            name = support_resist.loc[date, '종류']
            fig.add_trace(go.Scatter(
                x=[date + timedelta(dt) for dt in range(-20, 21)],
                y=[support_resist.loc[date, '레벨']] * 40,
                mode='lines',
                line=dict(color='blue' if name.startswith('저항선') else 'red', dash='dot', width=2),
                name='지지/저항선',
                legendgroup='지지/저항선',
                showlegend=False if n else True,
                visible='legendonly',
                hovertemplate=name + f'@{date.date()}<br>' + '가격:%{y:,}원<extra></extra>'
            ))
        
        # 필터
        for col in data.columns:
            if not (col.startswith('SMA') or col.startswith('IIR')):
                continue
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                customdata=self.format(date_list=data.index),
                visible='legendonly',
                hovertemplate=col + '<br>날짜:%{customdata}<br><extra></extra>'
            ))
        fig.update_layout(self.layout(title='주가 기본 분석',ytitle='가격[KRW]'))
        fig.update_layout(dict(
            legend=dict(traceorder='normal'),
            xaxis=dict(title='', showticklabels=True),
            xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='거래량', showgrid=True, gridcolor='lightgrey')
        ))
        if show:
            fig.show()
        if save:
            of.plot(fig, filename="chart-default.html", auto_open=False)
        return fig

    def guidance(self, frame:pd.DataFrame, by:str, show:bool=False, save:bool=False) -> go.Figure:
        """
        주가 가이던스(필터 선) 차트
        :param frame: [시가, 고가, 저가, 종가, SMA(n)D, EMA(n)D, FIR(n)D, IIR(n)D] 포함 프레임
        :param by: 시가/고가/저가/종가 중 택1
        :param show: True::즉시 Plot
        :param save: True::로컬 저장
        :return:
        """
        fig = go.Figure(layout=self.layout(title='주가 가이던스Guidance 차트', ytitle='가격(KRW)'))
        if '거래량' in frame.columns:
            frame.drop(columns=['거래량'], inplace=True)
        price = frame[['시가', '고가', '저가', '종가']].copy()
        guide = frame.drop(columns=price.columns).copy().join(price[by], how='left')

        for col in guide.columns:
            cond = (col[-3:] == '60D' or col == by) and (not col.startswith('FIR'))
            hover = col + ': %{y:,}원' if col == by else col + ': %{y:,.2f}'
            fig.add_trace(go.Scatter(
                x=guide.index,
                y=guide[col],
                name=col,
                visible=True if cond else 'legendonly',
                meta=self.format(date_list=guide.index),
                hovertemplate=hover + '<br>날짜: %{meta}<extra></extra>'
            ))
        fig.add_trace(
            go.Candlestick(
                x=price.index,
                customdata=self.format(date_list=price.index),
                open=price['시가'],
                high=price['고가'],
                low=price['저가'],
                close=price['종가'],
                increasing_line=dict(color='red'),
                decreasing_line=dict(color='blue'),
                name='일봉',
                visible='legendonly',
                showlegend=True,
            )
        )
        if show:
            fig.show()
        if save:
            of.plot(fig, filename="chart-guidance.html", auto_open=False)
        return fig

    def tendency(self, frame:pd.DataFrame, by:str, show:bool=False, save:bool=False) -> go.Figure:
        """
        주가 추세선 차트
        :param frame: [시가, 고가, 저가, 종가, *args]
        :param by: 시가/고가/저가/종가 중 택1
        :param show: True::즉시 Plot
        :param save: True::로컬 저장
        :return:
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if '거래량' in frame.columns:
            frame.drop(columns=['거래량'], inplace=True)
        price = frame[['시가', '고가', '저가', '종가']].copy()
        trend = frame.drop(columns=price.columns).copy()

        for col in trend.columns:
            if col.startswith('d'):
                continue
            fig.add_trace(
                go.Scatter(
                    x=trend.index,
                    y=trend[col],
                    customdata=self.format(date_list=trend.index),
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
                x=price.index,
                y=price[by],
                meta=self.format(date_list=price.index),
                name='종가',
                mode='lines',
                showlegend=True,
                visible=True,
                hovertemplate='날짜:%{meta}<br>' + by + ':%{y:,}원<extra></extra>',
            ),
            secondary_y=True
        )
        fig.update_layout(self.layout(
            title='주가 추세Trend 차트',
            ytitle='추세값[-]',
        ))
        fig.update_layout(yaxis=dict(zeroline=True, zerolinecolor='grey', zerolinewidth=1))
        if show:
            fig.show()
        if save:
            of.plot(fig, filename="chart-tendency.html", auto_open=False)
        return fig

    def momentum(self, frame:pd.DataFrame, show: bool = False, save: bool = False) -> go.Figure:
        """
        추세선 모멘텀 차트
        :param frame: [*args] 추세선/모멘텀 데이터
        :param show: True::즉시 Plot
        :param save: True::로컬 저장
        :return: 
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        frm = frame[60:].copy()
        cols = [col for col in frm.columns if col.startswith('d')]
        for col in cols:
            cond = (col.endswith('FIR') or col.endswith('IIR')) and '중기' in col
            fig.add_trace(
                go.Scatter(
                    x=frm.index,
                    y=frm[col],
                    name=col,
                    meta=self.format(date_list=frm.index),
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
        if save:
            of.plot(fig, filename="chart-momentum.html", auto_open=False)
        return fig

    def scatter(
            self,
            frame:pd.DataFrame, series:pd.Series=pd.DataFrame(),
            label:str='', threshold:float=0, td:int=20,
            show:bool=False, save:bool=False
    ):
        """
        지표 대비 수익률 상관관계 차트
        :param series: 지표 1-D
        :param frame: 지표와 수익률 정답지 포함 데이터프레임
        :param label: (frame 입력 시) 지표 열(Column) 이름
        :param threshold: 지표 기준 값
        :param td: 거래일
        :param show: True::즉시 Plot
        :param save: True::로컬 저장
        :return:
        """
        if not series.empty:
            label = '지표'
            series.name = label
            frame = frame.join(series, how='left')

        fig = go.Figure(layout=self.layout(
            title=f'{label}-수익률 산포도 :: ({frame.index[0].date()} ~ {frame.index[-1].date()})',
            xtitle=f'{label}',
            ytitle=f'{td}거래일 수익률[%]'
        ))
        fig.add_trace(
            go.Scatter(
                x=frame[label],
                y=frame[f'PERF-{td}TD'],
                mode='markers',
                marker=dict(color=frame[f'COLOR-{td}TD']),
                meta=self.format(date_list=frame.index),
                hovertemplate='날짜:%{meta}<br>' + label + ':%{x:.2f}<br>수익률:%{y:.2f}%<extra></extra>'
            )
        )
        cut = {10:3, 15:4, 20:5, 25:6}[td]
        zc = frame[frame[label] >= threshold].copy()
        ps = frame[frame[f'PERF-{td}TD'] > cut].copy()
        r_pass_zc = 100 * len(zc[zc[f'PERF-{td}TD'] > cut]) / len(zc)
        r_zc_pass = 100 * len(ps[ps[label] > threshold]) / len(ps)
        fig.update_layout(
            xaxis=dict(zerolinewidth=2, zerolinecolor='black'),
            yaxis=dict(zerolinewidth=2, zerolinecolor='black'),
            annotations=[
                dict(
                    text=f'ACHIEVE / ZERO-CROSS (%): {r_pass_zc:.2f}%<br>ZERO-CROSS / ACHIEVE (%): {r_zc_pass:.2f}%<br>',
                    showarrow=False,
                    xref="paper", yref="paper", x=1.0, y=1.0, align="left"
                )
            ],
        )
        if show:
            fig.show()
        if save:
            of.plot(fig, filename="chart-scatter.html", auto_open=False)
        return fig

class asset(toolkit):
    def __init__(self, **kwargs):
        """
        :param ticker: 종목코드
        :param kwargs:
        | KEYWORDS | TYPE |               COMMENT |                         POSSIBLE INPUTS |              DEFAULT |
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
        self._refer_ = pd.DataFrame()
        self._bound_ = pd.DataFrame()
        self._limit_ = pd.DataFrame()
        return

    @property
    def guide(self) -> pd.DataFrame:
        """
        주가 가이던스 (필터 선)
        :return:
        """
        if not self._guide_.empty:
            return self._guide_
        self._guide_ = self.guides(base=self.price[self.filterby], windows=self.windows)
        return self._guide_

    @property
    def trend(self) -> pd.DataFrame:
        """
        주가 추세선
        :return:
        """
        if not self._trend_.empty:
            return self._trend_
        self._trend_ = self.trends(base=self.price[self.filterby], windows=self.windows)
        return self._trend_

    @property
    def reference(self) -> pd.DataFrame:
        """
        과거 수익률 정답지
        :return:
        """
        if not self._refer_.empty:
            return self._refer_
        self._refer_ = self.__ans__(data=self.price, by='종가')
        return self._refer_

    @property
    def bound(self):
        """
        기간별 주가 진동 범위
        :return:
        """
        if not self._bound_.empty:
            return self._bound_
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
        self._bound_ = pd.concat(objs=objs, axis=1)
        return self._bound_

    @property
    def limit(self) -> pd.DataFrame:
        """
        지지선/저항선 표기
        :return:
        """
        if not self._limit_.empty:
            return self._limit_

        def is_support(df, i):
            _ = df['저가']
            support = _[i] < _[i - 1] < _[i - 2] and _[i] < _[i + 1]  < _[i + 2]
            return support

        def is_resistance(df, i):
            _ = df['고가']
            resistance = _[i] > _[i - 1] > _[i - 2] and _[i] > _[i + 1]  > _[i + 2]
            return resistance

        def is_far_from_level(l, s, lines):
            return np.sum([abs(l - x) < s for x in lines]) == 0

        frm = self.price[self.price.index >= (self.price.index[-1] - timedelta(180))].copy()
        s_hat = np.mean(frm['고가'] - frm['저가'])

        levels = []; index = []; types = []; s_cnt = 1; r_cnt = 1
        for n, date in enumerate(frm.index[2: len(frm) - 2]):
            if is_support(frm, n):
                if is_far_from_level(l = frm['저가'][n], s=s_hat, lines=levels):
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


class filters(toolkit):
    def response(self, price:pd.Series, date:datetime=None, filter_type:str='iir', gap:int=60) -> pd.DataFrame:
        """
        필터의 입력 날짜 안정화 기간 데이터프레임 :: IIR 필터 전용
        :param price: 필터 대상 가격
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
        base = price.copy()

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

    def answers(self, base:pd.Series, count:int=-1):
        """
        필터 안정화 추정 학습용 데이터 샘플 (정답지)
        :param base: 가격 정보
        :param count:
        :return:
        """
        time_span = base.index.tolist()
        samples = time_span[121:-120] if count == -1 else random.sample(time_span[121:-120], count)
        objs = []
        for sample in samples:
            raw = self.response(date=sample, filter_type='iir', gap=60)
            raw.drop(columns=[sample.strftime("%Y-%m-%d")], inplace=True)
            err = (100 * (raw.iloc[-1] / raw.iloc[0] - 1)).to_dict()
            objs.append(err)
        return pd.DataFrame(data=objs, index=samples)


if __name__ == "__main__":

    stock = asset(ticker='253450', src='offline')
    print(f"{stock.name}({stock.ticker})")

    display = chart(name=stock.name, ticker=stock.ticker)
    fig = display.default(
        base=pd.concat([stock.price, stock.guide, stock.bound], axis=1),
        support_resist=stock.limit,
        show=False, save=True
    )
    # fig = display.guidance(
    #     frame=pd.concat([stock.price, stock.guide], axis=1), by=stock.filterby,
    #     show=False, save=True
    # )
    # display.tendency(
    #     frame=pd.concat([stock.price, stock.trend], axis=1), by=stock.filterby,
    #     show=False, save=True
    # )
    # display.momentum(
    #     frame=pd.concat([stock.price, stock.trend], axis=1),
    #     show=False, save=True
    # )
    # display.scatter(
    #     frame=pd.concat([stock.reference, stock.trend], axis=1), label='d중기IIR',
    #     threshold=0, td=20, show=False, save=True
    # )

    # dater = datum(ticker='000660')
    # print(f"{dater.name}({dater.ticker})")
    # test_frame = dater.oscillation(date=None)
    # test_sample = dater.sampler(count=-1)
    # print(test_sample)

    # verify = myChart.verifier(myChart.price['저가'], pick='btr', sample=100)
    # print(verify)
    # print(verify.describe())
    # myChart.guidance_stability(pick='lpf', show=True, sample=datetime(2014, 5, 30))

