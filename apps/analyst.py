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
            # N, beta = kaiserord(ripple=50, width=5/(252 / 2))
            # N, beta = kaiserord(ripple=50, width=normal_cutoff)
            N, beta = kaiserord(ripple=8 if cutoff == 5 else cutoff, width=50/(252 / 2))
            taps = firwin(N, normal_cutoff, window=('kaiser', beta))
            objs[f'LPF{cutoff}D'] = pd.Series(data=lfilter(taps, 1.0, base), index=base.index)
        return pd.concat(objs=objs, axis=1)

    def verifier(self, base:pd.DataFrame, pick:str, sample) -> pd.DataFrame:
        """
        필터 안정화 기간 판정 프레임
        :param base: 필터 대상 시계열 데이터(주가)
        :param pick: 적용 필터 종류
        :param sample: 날짜(datetime) 입력 또는 정수(int) 입력
        :return:
        ** sample == 1 또는 sample == datetime(2015, 4, 3)
                   BTR5D        BTR10D        BTR20D        BTR60D   2015-04-03
        0   44000.011094  44014.566377  44126.798049  44595.287779        0일차
        1   44151.467641  44338.634013  44519.024210  44857.759991        1일차
        ...          ...           ...           ...           ...          ...
        28  44098.916008  44102.594753  44252.354188  44743.962672       28일차
        29  44098.916008  44102.594750  44252.280630  44715.876359       29일차

        ** sample > 1
                       BTR5D    BTR10D    BTR20D    BTR60D
        2013-06-28  0.048598  1.471260  2.824817  4.069192
        2015-05-07  0.846610  1.600933  2.043675  2.593163
        ...              ...       ...       ...       ...
        2020-07-24  0.273825  0.388971  0.254921  0.848624
        2020-03-11  1.484396  1.829621  2.249649  3.825485
        """
        if type(sample) == int and len(base.index) < (1.5 * sample):
            raise ValueError('Verification Failed :: Not Enough Data Samples (분석 종목을 변경하세요)')

        app = {'sma': self.__sma__, 'ema': self.__ema__, 'btr': self.__btr__, 'lpf': self.__lpf__}[pick.lower()]

        time_span = base.index[121:-30].tolist()
        time_pick = random.sample(time_span, sample) if type(sample) == int else sample
        def __proc__(date:datetime, add_col:bool) -> pd.DataFrame:
            n_date = time_span.index(date)
            datum = []
            for n, d in enumerate(time_span[n_date : n_date + 25]):
                rebase = base[base.index <= d].copy()
                out = app(base=rebase, windows=[5, 10, 20, 60])
                data = out.iloc[-(n+1)].to_dict()
                if add_col: data[f'{date.strftime("%Y-%m-%d")}'] = f'{n}일차'
                datum.append(data)
            return pd.DataFrame(data=datum)

        if sample == 1 or type(sample) == datetime:
            return __proc__(date=time_pick[0] if type(sample) == int else time_pick, add_col=True)

        objs = []
        for _t in time_pick:
            ver = __proc__(date=_t, add_col=False)
            err = (100 * (ver.iloc[-1]/ver.iloc[0] - 1)).abs().to_dict()
            objs.append(err)
        return pd.DataFrame(data=objs, index=time_pick)


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
        self.src = kwargs['src'] if 'src' in keys else 'online'
        self.windows = kwargs['windows'] if 'windows' in keys else [5, 10, 20, 60, 120]
        self.filterby = kwargs['filterby'] if 'filterby' in keys else '저가'

        self.name = self.meta.loc[self.ticker, '종목명']
        self.sector = self.meta.loc[self.ticker, '섹터']
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
        )

    def guidance(self, *args, show:bool=False) -> go.Figure:
        """
        주가 가이던스(필터 선) 차트
        :param args: Plot 대상 필터 (기본 :: 전체)
        :param show: True::즉시 Plot
        :return:
        """
        flg = True if not args else False
        fig = go.Figure(layout=self.layout(title='주가 가이던스 차트', ytitle='가격(KRW)'))
        frm = self.guide.copy()
        frm = frm.join(self.price[self.filterby], how='left')
        for col in frm.columns:
            if (not flg) and (not col == self.filterby):
                line = [l for l in ['SMA', 'EMA', 'BTR', 'LPF'] if col.startswith(l)][0]
                if (not flg) and (not line in args):
                    continue

            cond = col[-3:] == '20D' or col[-3:] == '60D' or (not col.endswith('D'))
            unit = '[-]' if col.endswith('D') else '원'
            fig.add_trace(go.Scatter(
                x=frm.index,
                y=frm[col],
                name=col,
                visible=True if cond else 'legendonly',
                meta=['{}/{}/{}'.format(d.year, d.month, d.day) for d in frm.index],
                hovertemplate=col + '<br>날짜: %{meta}<br>필터: %{y:,.2f}' + unit + '<extra></extra>'
            ))
        if show:
            fig.show()
        return fig

    def guidance_stability(self, pick:str, sample, show:bool=False) -> go.Figure:
        """
        주가 가이던스(필터 선) 임의의 날짜 안정화 기간 검증용 차트
        :param pick: Plot 대상 필터
        :param sample: 날짜(datetime) 입력 또는 정수(int) 입력
        :param show: True::즉시 Plot
        :return:
        """
        if type(sample) == int and sample > 1:
            raise ValueError('Only ONE Datetime Point must be passed :: 단일 시점 분석 대상임')

        frm = self.verifier(base=self.price[self.filterby], pick=pick.lower(), sample=sample)
        fig = go.Figure(layout=self.layout(
            title=f'{frm.columns[-1]} 안정 반응 기간',
            xtitle='순차 기간',
            ytitle=f'{self.filterby} 필터'
        ))

        for col in frm.columns[:-1]:
            fig.add_trace(go.Scatter(
                x=frm[frm.columns[-1]],
                y=frm[col].values,
                name=col,
                mode='lines+markers',
                meta=[100 * (val/frm[col].values[0] - 1) for val in frm[col].values],
                hovertemplate='일차:%{x}<br>값:%{y:,.2f}<br>오차:%{meta:.2f}%<extra></extra>',
                visible=True if col.endswith('20D') else 'legendonly'
            ))
        if show:
            fig.show()
        return fig



if __name__ == "__main__":

    myChart = chart(meta=stocks(), ticker='000660')
    myChart.guidance('BTR', 'SMA', 'LPF', show=True)

    # verify = myChart.verifier(myChart.price['저가'], pick='btr', sample=100)
    # print(verify)
    # print(verify.describe())
    # myChart.guidance_stability(pick='lpf', show=True, sample=datetime(2014, 5, 30))

