import os, random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter


__root__ = os.path.dirname(os.path.dirname(__file__))

def stocks(mode:str='in-use') -> pd.DataFrame:
    """
    종목 메타데이터 호출
    :return:
    """
    frm = pd.concat([
        pd.read_csv(
            'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/group/WI26.csv',
            encoding='utf-8',
            index_col='종목코드'
        ).drop(columns=['종목명']),
        pd.read_csv(
            'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/meta-stock.csv',
            encoding='utf-8',
            index_col='종목코드'
        ).drop(columns=['상장일', '거래소']).join(
            other=pd.read_csv(
                'http://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/market/market.csv',
                encoding='utf-8',
                index_col='종목코드'
            ).drop(columns=['종목명', '종가', '시가총액']),
            how='left'
        )],
        axis=1
    )

    if mode == 'in-use':
        frm = frm[
            ['섹터', '종목명', '종가', '시가총액', 'R1D', 'R1W', 'R1M', 'PER', 'DIV']
        ].rename(columns={'R1D': '1일등락', 'R1W': '1주등락', 'R1M': '1개월등락'})
        for col in frm.columns:
            if '등락' in col:
                frm[col] = round(frm[col], 2).astype(str) + '%'
        frm['PER'] = round(frm['PER'], 2)
        frm['종가'] = frm['종가'].apply(lambda p: '{:,}원'.format(int(p)))
        cap = (frm["시가총액"] / 100000000).astype(int).astype(str)
        frm['시가총액'] = cap.apply(lambda v: v + "억" if len(v) < 5 else v[:-4] + '조 ' + v[-4:] + '억')
    frm.index = frm.index.astype(str).str.zfill(6)
    return frm


class toolkit:

    @staticmethod
    def __sma__(base:pd.Series, windows:list) -> pd.DataFrame:
        """
        단순 이동 평균(Simple Moving Average) 필터 (FIR :: Not Use Feedback)
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
        지수 이동 평균(Exponent Moving Average) 필터 (FIR :: Not Use Feedback)
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :return:
        """
        return pd.concat(
            objs={f'EMA{win}D':base.ewm(span=win).mean() for win in windows},
            axis=1
        )

    @staticmethod
    def __btr__(base:pd.Series, windows:list, order:int=1) -> pd.DataFrame:
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
            objs[f'BTR{cutoff}D'] = pd.Series(data=filtfilt(coeff_a, coeff_b, base), index=base.index)
        return pd.concat(objs=objs, axis=1)

    @staticmethod
    def __lpf__(base:pd.Series, windows:list) -> pd.DataFrame:
        """
        scipy 패키지
        :param base: 필터 대상 시계열 데이터(주가)
        :param windows: 필터 대상 데이터 개수 모음
        :return:
        """
        objs = {}
        for cutoff in windows:
            normal_cutoff = (252 / cutoff) / (252 / 2)
            N, beta = kaiserord(ripple=50, width=50/(252 / 2))
            taps = firwin(N, normal_cutoff, window=('kaiser', beta))
            objs[f'LPF{cutoff}D'] = pd.Series(data=lfilter(taps, 1.0, base), index=base.index)
        return pd.concat(objs=objs, axis=1)

    def verify_span(self, base:pd.DataFrame, kind:str) -> pd.DataFrame:
        """

        :param base:
        :param kind:
        :return:
        """
        app = {'sma':self.__sma__, 'ema':self.__ema__, 'btr':self.__btr__, 'lpf':self.__lpf__}[kind.lower()]
        ver = pd.DataFrame()
        for date in base.index[121:]:
            rebase = base[base.index <= date].copy()
            out = app(base=rebase, windows=[5, 10, 20, 60])
            ver = ver.append(out.iloc[-1])
        cols = ver.columns.tolist()
        ver.rename(columns=dict(zip(cols, [col + '-T' for col in cols])), inplace=True)
        return ver

    def verify_point(self, base:pd.DataFrame, kind:str) -> pd.DataFrame:
        """

        :param base:
        :param kind:
        :return:
        """
        app = {'sma': self.__sma__, 'ema': self.__ema__, 'btr': self.__btr__, 'lpf': self.__lpf__}[kind.lower()]

        time_span = base.index[121:-50].tolist()
        time_pick = random.sample(time_span, 1)[0]
        n_start = time_span.index(time_pick)
        datum = []
        for n, date in enumerate(time_span[n_start : n_start+50]):
            rebase = base[base.index <= date].copy()
            out = app(base=rebase, windows=[5, 10, 20, 60])
            data = out.iloc[-(n+1)].to_dict()
            data[f'{time_pick.strftime("%Y-%m-%d")}'] = f'{n}일차'
            datum.append(data)
        return pd.DataFrame(data=datum)

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
        keys = kwargs.keys()
        self.meta = kwargs['meta']
        self.ticker = kwargs['ticker'] if 'ticker' in keys else '005930'
        self.name = self.meta.loc[self.ticker, '종목명']
        self.src = kwargs['src'] if 'src' in keys else 'online'
        self.windows = kwargs['windows'] if 'windows' in keys else [5, 10, 20, 60, 120]
        self.filterby = kwargs['filterby'] if 'filterby' in keys else '저가'

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

        self.sma = self.__sma__(base=self.price[self.filterby], windows=self.windows)
        self.ema = self.__ema__(base=self.price[self.filterby], windows=self.windows)
        self.btr = self.__btr__(base=self.price[self.filterby], windows=self.windows, order=1)
        self.lpf = self.__lpf__(base=self.price[self.filterby], windows=self.windows)

        self._guide_ = pd.DataFrame()
        return

    @property
    def guide(self) -> pd.DataFrame:
        """
        주가 가이던스 (필터 선)
        :return:
        """
        if self._guide_.empty:
            self._guide_ = pd.concat(objs=[self.sma, self.ema, self.btr, self.lpf], axis=1)
        return self._guide_

class chart(asset):
    def layout(self, title:str='', xtitle:str='날짜', ytitle:str='') -> go.Layout:
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

    def filters(self, **kwargs):
        """
        주가 가이던스(필터 선) 차트
        :param kwargs:
        :return:
        """
        keys = kwargs.keys()
        pick = kwargs['pick'] if 'pick' in keys else None
        td = kwargs['td'] if 'td' in keys else ['20D', '60D']
        show = kwargs['show'] if 'show' in keys else False

        fig = go.Figure(layout=self.layout())
        frm = self.guide.copy()
        frm = frm.join(self.price[self.filterby], how='left')
        for col in frm.columns:
            if col == self.filterby or pick.lower() == 'all':
                pass
            else:
                line = [l for l in ['SMA', 'EMA', 'BTR', 'LPF'] if col.startswith(l)][0]
                if pick and (not line in pick):
                    continue

            unit = '[-]' if col.endswith('D') else '[KRW]'
            fig.add_trace(
                go.Scatter(
                    x=frm.index,
                    y=frm[col],
                    name=col,
                    visible=True if col[-3:] in td or (not col.endswith('D')) else 'legendonly',
                    meta=['{}/{}/{}'.format(d.year, d.month, d.day) for d in frm.index],
                    hovertemplate=col + '<br>날짜: %{meta}<br>필터: %{y:,.2f}' + unit + '<extra></extra>'
                )
            )
        if show:
            fig.show()
            return
        else:
            return fig

    def stability(self, kind:str, td:str='20D'):
        verify = self.verify_point(base=self.price[self.filterby], kind=kind.lower())
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=verify[verify.columns[-1]],
                y=verify[f'{kind.upper()}{td}'],
                name=f'{kind.upper()}{td}'
            )
        )
        fig.show()
        return



if __name__ == "__main__":

    myChart = chart(meta=stocks(), ticker='000660')

    verify = myChart.verify_point(base=myChart.price['종가'], kind='btr')

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=verify[verify.columns[-1]],
            y=verify['BTR20D'],
            name='BTR20D'
        )
    )
    fig.show()

    # myChart.filters(show=True, pick=['LPF', 'BTR'])