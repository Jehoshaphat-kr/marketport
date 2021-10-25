import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.signal import butter,filtfilt


__root__ = os.path.dirname(os.path.dirname(__file__))
# ================================================================================================================== #
#                                               기초 함수 Basic Functions                                            #
# ================================================================================================================== #
def stocks() -> pd.DataFrame:
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
            )[['R1D', 'R1W', 'R1M', 'PER', 'DIV']].rename(columns={'R1D': '1일등락', 'R1W': '1주등락', 'R1M': '1개월등락'}),
            how='left'
        )],
        axis=1
    )
    for col in frm.columns:
        if '등락' in col:
            frm[col] = round(frm[col], 2).astype(str) + '%'
    frm['PER'] = round(frm['PER'], 2)
    frm['종가'] = frm['종가'].apply(lambda p: '{:,}원'.format(int(p)))
    cap = (frm["시가총액"] / 100000000).astype(int).astype(str)
    frm['시가총액'] = cap.apply(lambda v: v + "억" if len(v) < 5 else v[:-4] + '조 ' + v[-4:] + '억')
    frm.index = frm.index.astype(str).str.zfill(6)
    return frm

def indices(mode:str='display') -> pd.DataFrame:
    """
    인덱스 메타데이터 호출
    :param mode:
    :return:
    """
    index_raw = pd.read_csv(
        'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/meta-index.csv',
        encoding='utf-8',
        index_col='종목코드'
    )
    objs = []
    for t, name in [('KS', '코스피'), ('KQ', '코스닥'), ('KX', 'KRX'), ('TM', '테마')]:
        obj = index_raw[index_raw['거래소'] == t].copy()
        obj.index.name = f'{name}코드'
        obj.rename(columns={'종목명': f'{name}종류'}, inplace=True)
        obj.drop(columns=['거래소'], inplace=True)
        obj.reset_index(level=0, inplace=True)
        objs.append(obj)
    frm = pd.concat(objs=objs, axis=1).fillna('-')
    return frm if mode=='display' else index_raw

def calc_filtered(data: pd.Series, window_or_cutoff:list) -> pd.DataFrame:
    """
    이동평균선 / 저대역통과선 프레임
    :param data: 시가/저가/고가/종가 중
    :param window_or_cutoff:
    :return:
    """
    def __lpf__(cutoff: int, sample: int = 252, order: int = 1) -> pd.Series:
        """
        Low Pass Filter
        :param cutoff: 컷 오프 주파수
        :param sample: 기저 주파수
        :param order: 필터 차수
        :return:
        """
        normal_cutoff = (sample / cutoff) / (sample / 2)
        coeff_a, coeff_b = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(coeff_a, coeff_b, data)
        return pd.Series(data=y, index=data.index)

    mafs = pd.concat(
        objs={f'MAF{str(window).zfill(2)}D': data.rolling(window).mean() for window in window_or_cutoff},
        axis=1
    )
    lpfs = {f'LPF{str(cutoff).zfill(2)}D': __lpf__(cutoff=cutoff, sample=252) for cutoff in window_or_cutoff}
    return mafs.join(other=pd.concat(objs=lpfs, axis=1), how='left')

def calc_yield(data: pd.Series) -> pd.DataFrame:
    """
    1개월:5년 누적 수익률
    :param data: 시가/저가/고가/종가 중
    :return:
    """
    toc = data.index[-1]
    _5y = 100 * ((data.pct_change().fillna(0) + 1).cumprod() - 1)
    _3y = 100 * ((data[data.index >= (toc - timedelta(365 * 3))].pct_change().fillna(0) + 1).cumprod() - 1)
    _1y = 100 * ((data[data.index >= (toc - timedelta(365))].pct_change().fillna(0) + 1).cumprod() - 1)
    ytd = 100 * ((data[data.index >= datetime(toc.year, 1, 2)].pct_change().fillna(0) + 1).cumprod() - 1)
    _6m = 100 * ((data[data.index >= (toc - timedelta(183))].pct_change().fillna(0) + 1).cumprod() - 1)
    _3m = 100 * ((data[data.index >= (toc - timedelta(91))].pct_change().fillna(0) + 1).cumprod() - 1)
    _1m = 100 * ((data[data.index >= (toc - timedelta(30))].pct_change().fillna(0) + 1).cumprod() - 1)
    return pd.concat(
        objs={'1개월': _1m, '3개월': _3m, '6개월': _6m, 'YTD': ytd, '1년': _1y, '3년': _3y, '5년': _5y},
        axis=1
    )

def fetch_finance(ticker:str) -> (pd.DataFrame, pd.DataFrame):
    """
    재무제표 기본형 다운로드
    :param ticker: 종목코드
    :return:
    """
    def __form__(arg, key):
        if key == '개요':
            cols = arg.columns.tolist()
            arg.set_index(keys=[cols[0]], inplace=True)
            arg.index.name = None
            arg.columns = arg.columns.droplevel()
            arg = arg.T
            return arg

        if key == '항목':
            arg.set_index(keys=['항목'], inplace=True)
            arg.index.name = None
            arg = arg.T
            return arg

        if key == '연구개발':
            arg.set_index(keys=['회계연도'], inplace=True)
            arg.index.name = None
            arg = arg[['R&D 투자 총액 / 매출액 비중.1', '무형자산 처리 / 매출액 비중.1', '당기비용 처리 / 매출액 비중.1']]
            arg = arg.rename(columns={'R&D 투자 총액 / 매출액 비중.1': 'R&D투자비중',
                                      '무형자산 처리 / 매출액 비중.1': '무형자산처리비중',
                                      '당기비용 처리 / 매출액 비중.1': '당기비용처리비중'})
            return arg

        if key == '인원':
            arg.set_index(keys=['회계연도'], inplace=True)
            arg.index.name = None
            arg = arg[['구분', '기말인원(총계)', '평균근속연수']]
            arg = arg.loc[arg['구분'] == '총계']
            arg = arg[['기말인원(총계)', '평균근속연수']]
            return arg

        if key == '재무':
            cols = arg.columns.tolist()
            arg.set_index(keys=[cols[0]], inplace=True)
            arg.index.name = None
            if "전년동기" in arg.columns.values:
                del arg['전년동기']
                del arg['전년동기(%)']
            arg = arg.T
            return arg
        return

    ''' 
    1. 기업 개요
    연결 제무제표 또는 별도 제무제표 유효성 판정
    매출 추정치 존재 유무로 판정 (매출 추정치 존재 시 유효 판정)
    i.11 = 연간 연결 제무제표
    i.14 = 연간 별도 제무제표
    '''
    link = "http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701"
    table = pd.read_html(link % ticker, encoding='utf-8')
    if table[11].iloc[0].isnull().sum() > table[14].iloc[0].isnull().sum():
        a_table = table[14]
        q_table = table[15]
    else:
        a_table = table[11]
        q_table = table[12]
    annuals = [__form__(a_table, key='개요')]
    quarters = [__form__(q_table, key='개요')]

    ''' 2. 기업 관리 '''
    link = "http://comp.fnguide.com/SVO2/ASP/SVD_Corp.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=&NewMenuID=102&stkGb=701"
    table = pd.read_html(link % ticker, encoding='utf-8')
    annuals.append(__form__(table[6], key='항목'))  # 판관비
    annuals.append(__form__(table[7], key='항목'))  # 매출원가
    annuals.append(__form__(table[8], key='연구개발'))  # R&D투자
    quarters.append(__form__(table[9], key='인원'))  # 인원현황

    ''' 3. 재무 상태 '''
    # link = "http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A%s&cID=&MenuYn=Y&ReportGB=&NewMenuID=103&stkGb=701"
    # table = pd.read_html(link % self.ticker, encoding='utf-8')
    # for a in [2, 4]:
    #     annuals.append(self.__form__(table[a], key='재무'))
    # for q in [3, 5]:
    #     quarters.append(self.__form__(table[q], key='재무'))
    return pd.concat(annuals, axis=1), pd.concat(quarters, axis=1)
# ================================================================================================================== #


# ================================================================================================================== #
#                                                   분석 클래스 Classes                                              #
# ================================================================================================================== #
class frame:
    """
    기본 종목 분석 클래스
    :: basis: 시가 / 고가 / 저가 / 종가 / 거래량
    :: guideline: 이동평균선 / 저대역통과선(LPF)
    :: yieldline: 기간별 수익률
    :: trendline: 추세선
    :: momentum: 추세 미분선
    :: dropline: 낙폭
    :: finance: 재무제표 데이터
    """
    def __init__(self, ticker:str, on:str='종가', time_stamp:int=5, mode:str='offline'):
        """
        marketport @GITHUB 주가/지수 데이터 분석
        :param ticker: 종목코드/지수코드
        :param on: 가이드라인/모멘텀선/수익률곡선/낙폭 계산 시 참조 가격 (종가 기준)
        :param time_stamp: 시계열 Cut 기준 연수(year)
        :param mode: offline - 로컬 사용 / online - GITHUB 서버 사용
        """
        self.ticker = ticker
        self.key = on

        mkind = 'stock' if len(ticker) == 6 else 'index'
        meta = stocks() if mkind == 'stock' else indices(mode='raw')
        self.equity = meta.loc[ticker, '종목명']

        self.basis = pd.read_csv(
            f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{ticker}.csv',
            encoding='utf-8',
            index_col='날짜'
        ) if mode == 'online' else pd.read_csv(
            os.path.join(__root__, f'warehouse/series{ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        self.basis.index = pd.to_datetime(self.basis.index)
        if time_stamp:
            self.tic = self.basis.index[-1] - timedelta(365 * time_stamp)
            self.basis = self.basis[self.basis.index >= self.tic]

        self.g_line = pd.DataFrame()    # 필터선
        self.y_line = pd.DataFrame()    # 수익선
        self.t_line = pd.DataFrame()    # 추세선
        self.m_line = pd.DataFrame()    # 모멘텀
        self.d_line = pd.DataFrame()    # 낙폭
        self.f_a = pd.DataFrame()
        self.f_q = pd.DataFrame()
        return

    @property
    def guideline(self) -> pd.DataFrame:
        """
        이동평균선/노이즈제거선
        :return:
        """
        if not self.g_line.empty:
            return self.g_line

        self.g_line = calc_filtered(
            data=self.basis[self.key],
            window_or_cutoff=[5, 10, 20, 60, 120]
        )
        return self.g_line

    @property
    def yieldline(self) -> pd.DataFrame:
        """
        기간별 누적 수익률
        :return:
        """
        if not self.y_line.empty:
            return self.y_line

        self.y_line = calc_yield(data=self.basis[self.key])
        return self.y_line

    @property
    def trendline(self) -> pd.DataFrame:
        """
        주가/지수 추세선
        :return:
        """
        if not self.t_line.empty:
            return self.t_line

        guideline = self.guideline.copy()
        norm = 100 * (guideline - guideline.min().min())/(guideline.max().max() - guideline.min().min())

        objs = {
            '장기추세': norm['LPF120D'] - norm['MAF120D'],
            '중장기추세': norm['LPF60D'] - norm['MAF120D'],
            '중기추세': norm['LPF60D'] - norm['MAF60D'],
            '중단기추세': norm['LPF20D'] - norm['MAF60D'],
            '단기추세': norm['LPF05D'] - norm['MAF20D'],
            '중장기GC': norm['MAF60D'] - norm['MAF120D'],
            '중기GC': norm['MAF20D'] - norm['MAF60D'],
            '단기GC': norm['MAF05D'] - norm['MAF20D']
        }
        self.t_line = pd.concat(objs=objs, axis=1).dropna()
        return self.t_line

    @property
    def momentum(self) -> pd.DataFrame:
        """
        주가 모멘텀
        :return:
        """
        if not self.m_line.empty:
            return self.m_line

        base = self.trendline.copy()
        objs = {}
        for c in base.columns:
            if not '추세' in c:
                continue
            objs[c.replace('추세', '변화량')] = base[c].diff()
            objs[c.replace('추세', '모멘텀')] = base[c].diff().diff()
        self.m_line = pd.concat(objs=objs, axis=1)
        return self.m_line

    @property
    def drawdown(self) -> pd.DataFrame:
        """
        기간별 변동성
        :return:
        """
        if not self.d_line.empty:
            return self.d_line

        data = self.basis[self.key].copy()
        ytd = (datetime.today() - datetime(datetime.today().year, 1, 2)).days
        toc = data.index[-1]
        objs = {}
        for c, span in [
            ('1개월', 30), ('3개월', 91), ('6개월', 183), ('YTD', ytd), ('1년', 365), ('3년', 365 * 3), ('5년', 365 * 5)
        ]:
            src_dd = data[data.index >= (toc - timedelta(span))]
            src_dd = 100 * (src_dd / src_dd.cummax()).sub(1)

            objs[f'{c}낙폭'] = src_dd
        self.d_line = pd.concat(objs=objs, axis=1)
        return self.d_line

    @property
    def finance(self) -> (pd.DataFrame, pd.DataFrame):
        """
        재무제표 기본형
        :return:
        """
        if not self.f_a.empty and not self.f_q.empty:
            return self.f_a, self.f_q
        return fetch_finance(ticker=self.ticker)


class vstock(frame):
    """
    분석 시각화 툴
    """
    def show_price(self) -> go.Figure:
        """
        단일 종목 가격/지수 그래프
        :return:
        """
        xtick = '주가' if len(self.ticker) == 6 else '지수'
        fig = go.Figure()

        src = self.guideline
        for col in src.columns:
            fig.add_trace(
                go.Scatter(
                    x=src.index,
                    y=src[col],
                    name=col,
                    mode='lines',
                    line=dict(dash='dot' if 'MAF' in col else 'dash'),
                    showlegend=True,
                    visible=True if col in ['MAF120D', 'MAF60D','MAF20D', 'LPF60D', 'LPF05D'] else 'legendonly',
                    hovertemplate=f'{col}<extra></extra>'
                )
            )

        src = self.basis
        dform = ['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index]
        for col in ['종가', '시가', '저가', '고가']:
            fig.add_trace(
                go.Scatter(
                    x=src.index,
                    y=src[col],
                    customdata=dform,
                    name=col,
                    mode='lines',
                    showlegend=True,
                    visible=True if col == '종가' else 'legendonly',
                    hovertemplate='날짜:%{customdata}<br>' + col + ':%{y:,}원<extra></extra>',
                )
            )

        fig.add_trace(
            go.Candlestick(
                x=src.index,
                customdata=dform,
                open=src['시가'],
                high=src['고가'],
                low=src['저가'],
                close=src['종가'],
                increasing_line=dict(color='red'),
                decreasing_line=dict(color='blue'),
                name='일봉',
                visible='legendonly',
                showlegend=True,
            )
        )

        fig.update_layout(
            dict(
                title=f'<b>[{self.equity}({self.ticker})]</b> {xtick} 분석',
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
                yaxis=dict(title=f'{xtick}[KRW]', showgrid=True, zeroline=False, showticklabels=True, autorange=True, gridcolor='lightgrey'),
                xaxis=dict(
                    title='날짜', showgrid=True, zeroline=False, showticklabels=True, autorange=True, gridcolor='lightgrey',
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                ),
                xaxis_rangeslider=dict(visible=False)
            )
        )
        fig.update_traces(
            selector=dict(type='candlestick'),
            xhoverformat='%Y-%m-%d',
            yhoverformat=','
        )
        return fig

    def show_trend(self) -> go.Figure:
        """
        추세선 그래프
        :return:
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        src = self.trendline.copy()
        dform = ['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index]
        for col in src.columns:
            fig.add_trace(
                go.Scatter(
                    x=src.index,
                    y=src[col],
                    customdata=dform,
                    name=col,
                    mode='lines',
                    showlegend=True,
                    visible='legendonly' if col.endswith('GC') else True,
                    hovertemplate=col + '<br>추세값:%{y:.2f}<br>날짜:%{customdata}<br><extra></extra>',
                ),
                secondary_y=False
            )
        src = self.basis
        fig.add_trace(
            go.Scatter(
                x=src.index,
                y=src['종가'],
                meta=['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index],
                name='종가',
                mode='lines',
                showlegend=True,
                visible=True,
                hovertemplate='날짜:%{meta}<br>종가:%{y:,}원<extra></extra>',
            ),
            secondary_y=True
        )

        fig.update_layout(
            dict(
                title=f'<b>[{self.equity}({self.ticker})]</b> 추세선 분석',
                plot_bgcolor='white',
                annotations=[
                    dict(
                        text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )
                ],
                yaxis=dict(
                    showgrid=True, zeroline=True, showticklabels=True, autorange=True,
                    gridcolor='lightgrey', zerolinecolor='grey', zerolinewidth=2
                ),
                xaxis=dict(
                    title='날짜', showgrid=True, zeroline=False, showticklabels=True, autorange=True, gridcolor='lightgrey',
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                ),
                xaxis_rangeslider=dict(visible=False)
            )
        )
        fig.update_yaxes(title_text="정규값[-]", secondary_y=False)
        fig.update_yaxes(title_text="종가[KRW]", secondary_y=True)
        return fig

    def show_momentum(self) -> go.Figure:
        """
        모멘텀
        :return:
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        raw = self.momentum.copy().dropna()
        for col in raw.columns:
            src = raw[col]
            fig.add_trace(
                go.Scatter(
                    x=src.index,
                    y=src,
                    name=col,
                    meta=['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index],
                    hovertemplate=col + '<br>날짜: %{meta}<br>값:%{y:.4f}<extra></extra>'
                ),
                row=1 if '변화량' in col else 2, col=1
            )

        fig.update_layout(
            title=f'<b>[{self.equity}({self.ticker})]</b> 모멘텀 분석',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            yaxis=dict(
                title='정규값[-]', showgrid=True, zeroline=True, showticklabels=True, autorange=True,
                gridcolor='lightgrey', zerolinecolor='grey', zerolinewidth=1
            ),
            yaxis2=dict(
                title='정규값[-]', showgrid=True, zeroline=True, showticklabels=True, autorange=True,
                gridcolor='lightgrey', zerolinecolor='grey', zerolinewidth=1
            ),
            xaxis=dict(
                showgrid=True, zeroline=False, showticklabels=False, autorange=True, gridcolor='lightgrey',
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            ),
            xaxis2=dict(
                title='날짜', zeroline=False, showticklabels=True, autorange=True, gridcolor='lightgrey'
            )
        )
        return fig

    def show_drawdown(self) -> go.Figure:
        """
        변동성 및 낙폭
        :return:
        """

        fig = go.Figure()
        buttons = []

        riskline = self.drawdown
        for i, col in enumerate(riskline.columns):
            src = riskline[col].dropna()
            fig.add_trace(
                go.Scatter(
                    x=src.index,
                    y=src,
                    name=col,
                    meta= ['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index],
                    visible=True if '1년' in col else False,
                    hovertemplate=col + '<br>날짜: %{meta}<br>낙폭: %{y:.2f}%<extra></extra>'
                ),
            )
            gap = col.replace('낙폭', '')
            visible = [False] * len(riskline.columns)
            visible[i] = True
            buttons.append(dict(
                label=gap,
                method='update',
                args=[{'visible':visible}]
            ))

        fig.update_layout(dict(
            title=f'<b>[{self.equity}({self.ticker})]</b> 낙폭 분석',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            updatemenus=[dict(active=4, buttons=buttons, xanchor='right', yanchor='top', x=1, y=1.02)],
            yaxis=dict(title='낙폭[%]', showgrid=True, showticklabels=True, autorange=True, gridcolor='lightgrey'),
            xaxis=dict(title='날짜', showgrid=True, showticklabels=True, autorange=True, gridcolor='lightgrey'),
        ))
        return fig

    def show_sales(self, kind:str='annual') -> go.Figure:
        """
        매출/영업이익/당기순이익
        :param kind:
        :return:
        """
        t_kind = '연간' if kind == 'annual' else '분기'
        fig = go.Figure()

        raw = self.finance[0] if kind == 'annual' else self.finance[1]
        key = '매출액'
        key = '순영업수익' if '순영업수익' in raw.columns else key
        key = '보험료수익' if '보험료수익' in raw.columns else key

        for col in [key, '영업이익', '당기순이익']:
            src = raw[col].dropna()
            src = src.astype(int)
            fig.add_trace(
                go.Bar(
                    x=src.index,
                    y=src,
                    name=col,
                    meta=[str(v) + '억원' if v < 10000 else str(v)[:-4] + '조 ' + str(v)[-4:] + '억원' for v in src],
                    opacity=0.9,
                    hovertemplate=col + '<br>%{meta}<extra></extra>'
                )
            )

        fig.update_layout(dict(
            title=f'<b>[{self.equity}({self.ticker})]</b> {t_kind} 영업 실적 분석',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(x=0, y=1.0, bgcolor='white', bordercolor='white'),
            yaxis=dict(
                title='KRW [억원]', showgrid=True, zeroline=True, showticklabels=True, autorange=True, gridcolor='lightgrey'
            ),
            xaxis=dict(title=t_kind, showgrid=False, zeroline=False, showticklabels=True, autorange=True)
        ))
        return fig

    def show_financial_ratio(self, kind='annual') -> go.Figure:
        """
        매출/영업이익/당기순이익 성장율
        :param kind:
        :return:
        """
        t_kind = '연간' if kind == 'annual' else '분기'
        fig = go.Figure()

        raw = self.finance[0] if kind == 'annual' else self.finance[1]
        for col in ['ROA', 'ROE', '영업이익률', '배당수익률']:
            if col == '배당수익률' and t_kind == '분기':
                continue
            src = raw[col].dropna()
            fig.add_trace(
                go.Scatter(
                    x=src.index,
                    y=src,
                    name=col,
                    mode='lines+markers',
                    opacity=0.9,
                    hovertemplate=col + '<br>%{y:.2f}%<extra></extra>'
                )
            )

        fig.update_layout(dict(
            title=f'<b>[{self.equity}({self.ticker})]</b> {t_kind} 수익성 분석',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            legend=dict(x=0, y=1.0, bgcolor='white', bordercolor='white'),
            yaxis=dict(
                title='비율[%]', showgrid=True, zeroline=True, showticklabels=True, autorange=True,
                gridcolor='lightgrey'
            ),
            xaxis=dict(title=t_kind, showgrid=False, zeroline=False, showticklabels=True, autorange=True)
        ))
        return fig


class evaluate(frame):
    """
    종목 추세 평가 모델
    """
    @property
    def modelInYoung(self) -> pd.DataFrame:
        """
        최초 모델 ::: <평가지표:기본모델>@Google Drive
        :return:
        """
        price = self.basis['종가'].copy().values
        guide = self.guideline.copy()
        trend = self.trendline.copy()
        deriv = self.momentum.copy()

        objs = {'종목코드': self.ticker, '종목명': self.equity, 'MA상회': 0}
        if len(price) < 252 * 2:
            return pd.DataFrame(data=objs, index=[1])

        ''' A '''
        w10 = [7.5, 7, 6.5, 6, 5.5, 4.5, 4, 3.5, 3, 2.5]
        for col, term in [('장기', 120), ('중기', 60)]:
            score = [0 if price[-(n + 1)] < guide[f'MAF{term}D'].values[-(n + 1)] else w for n, w in enumerate(w10)]
            objs['MA상회'] += sum(score)

        ''' B '''
        w10 = [15, 14, 13, 12, 11, 9, 8, 7, 6, 5]
        w05 = [24, 22, 20, 18, 16]
        for term, wt, score in [('중장기', w10, 100), ('중기', w10, 100), ('단기', w05, 100)]:
            fails = [0 if deriv[f'{term}변화량'].values[-(n + 1)] > 0 else w for n, w in enumerate(wt)]
            objs[f'{term}추세'] = round(score - sum(fails), 2)

        ''' C '''
        wL = [17.6, 19.2, 16, 14.4, 12.8]
        wM = [22, 24, 20, 18, 16]
        wS = [26.4, 28.8, 24, 21.6, 19.2]
        for term, wt, score in [('중장기', wL, 80), ('중기', wM, 100), ('단기', wS, 120)]:
            fails = [0 if deriv[f'{term}모멘텀'].values[-(n + 1)] > 0 else w for n, w in enumerate(wt)]
            objs[f'{term}모멘텀'] = round(score - sum(fails), 2)

        ''' D '''
        wL = [12, 11.2, 10.4, 9.6, 8.8, 7.2, 6.4, 5.6, 4.8, 4]
        wM = [15, 14, 13, 12, 11, 9, 8, 7, 6, 5]
        wS = [26.4, 28.8, 24, 21.6, 19.2]
        for term, wt, score in [('중장기', wL, 80), ('중기', wM, 100), ('단기', wS, 120)]:
            fails = [0 if trend[f'{term}추세'].values[-(n + 1)] > 0 else w for n, w in enumerate(wt)]
            objs[f'{term}크로스오버'] = round(score - sum(fails), 2)

        objs['총점'] = round(sum([value for key, value in objs.items() if not key in ['종목코드', '종목명']]), 2)

        ''' E '''
        wt = [60, 70, 50]
        short = trend['단기추세'].values
        for n, w in enumerate(wt):
            curr = short[-(n + 1)]
            prev = short[-(n + 2)]
            if curr > prev and curr * prev < 0:
                objs['총점'] += w
        return pd.DataFrame(data=objs, index=[1])

    @property
    def modelJeMyoung(self) -> pd.DataFrame:
        price = self.basis.종가.values
        guide = self.guideline.copy()
        trend = self.trendline.copy()
        deriv = self.momentum.copy()
        if len(price) < 252 * 2:
            return pd.DataFrame(data={'종목코드':self.ticker, '종목명': self.equity}, index=[1])

        table = pd.DataFrame(
            data={
                '종목코드':self.ticker, '종목명': self.equity,
                '과락여부': 'PASS',
                'MA상회':0, '크로스오버':0,

                '장기추세상승':'N/A',
                '단기상승반영':'N/A',
                '종합': 0
            },
            index=[0]
        )

        ''' 과락 여부 판단 '''
        trend_l = trend['중장기추세'].values
        deriv_l = deriv['중장기변화량'].values
        deriv_m = deriv['중기변화량'].values
        for td in range(-10, 0, 1):
            if (trend_l[td] < 0) or ( (td >= -5) and (deriv_l[td] < 0) or (deriv_m[td] < 0) ):
                table.loc[0, '과락여부'] = 'FAIL'
                return table

        ''' A '''
        w10 = [7.5, 7, 6.5, 6, 5.5, 4.5, 4, 3.5, 3, 2.5]
        for col, term in [('장기', 120), ('중기', 60)]:
            score = [0 if price[-(n + 1)] < guide[f'MAF{term}D'].values[-(n + 1)] else w for n, w in enumerate(w10)]
            table.loc[0, 'MA상회'] += sum(score)

        ''' B '''

        # s_trend = trend['단기추세'].values
        # if s_trend[-1] > 0 and s_trend[-2] > 0 and s_trend[-3] and s_trend[-4]:
        #     table.loc[0, '단기상승반영'] = 'FAIL'
        #
        # s_deriv = deriv['단기변화량'].values
        # for n in range(4):
        #     if s_deriv[-(n+1)] < 0:
        #         table.loc[0, '단기추세상승'] = 'FAIL'
        #         break
        #
        # m_trend = trend['중기추세'].values
        # m_deriv = deriv['중기변화량'].values
        # m_mtum = deriv['중기모멘텀'].values
        # for n in range(10):
        #     if (n < 2 and m_mtum[-(n+1)] < 0) or (n < 5 and m_trend[-(n+1)] < 0) or (m_deriv[-(n+1)] < 0):
        #         table.loc[0, '중기추세상승'] = 'FAIL'
        #         break
        #
        # l_trend = trend['중장기추세'].values
        # l_deriv = deriv['중장기변화량'].values
        # for n in range(10):
        #     if l_trend[-(n+1)] < 0 or l_deriv[-(n+1)] < 0:
        #         table.loc[0, '중기추세상승'] = 'FAIL'
        #         break

        table.loc[0, '종합'] = table[['MA상회']].loc[0].sum()
        return table


if __name__ == "__main__":
    # print(stocks())
    # print(indices())

    asset = frame(ticker='2203')
    # print(asset.equity)
    # print(asset.basis)
    # print(asset.guideline)
    # print(asset.yieldline)
    # print(asset.trendline)
    # print(asset.momentum)

    # evaluation = evaluate(ticker='051910')
    # print(evaluation.modelInYoung.T)
    # print(evaluation.modelJeMyoung.T)

    # display = vstock(ticker='003670')
    # display.show_price().show()
    # display.show_trend().show()
    # display.show_momentum().show()
    # display.show_drawdown().show()
    # display.show_sales(kind='annual').show()
    # display.show_sales(kind='quarter').show()
    # display.show_financial_ratio(kind='annual').show()
    # display.show_financial_ratio(kind='quarter').show()
