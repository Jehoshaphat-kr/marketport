import os, random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.signal import butter, kaiserord, firwin, filtfilt, lfilter

__root__ = os.path.dirname(os.path.dirname(__file__))
# ================================================================================================================== #
#                                               기초 함수 Basic Functions                                            #
# ================================================================================================================== #
def stocks(mode :str ='in-use') -> pd.DataFrame:
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

def indices(mode :str ='display') -> pd.DataFrame:
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
    return frm if mode == 'display' else index_raw

def calc_filtered(data: pd.Series, window_or_cutoff :list, mode :str ='lowpass') -> pd.DataFrame:
    """
    이동평균선 / 저대역통과선 프레임
    :param data: 시가/저가/고가/종가 중
    :param window_or_cutoff:
    :param mode: 'butter', 'lowpass'
    :return:
    """
    def __btf__(cutoff: int, order: int = 1) -> pd.Series:
        """
        Low Pass Filter
        :param cutoff: 컷 오프 주파수
        :param sample: 기저 주파수
        :param order: 필터 차수
        :return:
        """
        normal_cutoff = (252 / cutoff) / (252 / 2)
        coeff_a, coeff_b = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(coeff_a, coeff_b, data)
        return pd.Series(data=y, index=data.index)

    def __lpf__(cutoff):
        normal_cutoff = (252 / cutoff) / (252 / 2)
        x = data.values
        y = [0] * len(data)
        yk = x[0]
        for k in range(len(data)):
            yk += normal_cutoff * (x[k] - yk)
            y[k] = yk
        return pd.Series(data=y, index=data.index)

    apps = __btf__ if mode == 'butter' else __lpf__
    mafs = pd.concat(
        objs={f'MAF{str(window).zfill(2)}D': data.rolling(window).mean() for window in window_or_cutoff},
        axis=1
    )
    lpfs = {f'LPF{str(cutoff).zfill(2)}D': apps(cutoff=cutoff) for cutoff in window_or_cutoff}
    return mafs.join(other=pd.concat(objs=lpfs, axis=1), how='left')

def calc_trending(data: pd.DataFrame) -> pd.DataFrame:
    """
    필터선 기준 추세/변화량/모멘텀 정보 생성
    :param data:
    :return:
    """
    norm = 100 * (data - data.min().min()) / (data.max().max() - data.min().min())
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
    _df_ = pd.concat(objs=objs, axis=1)

    objs = {}
    for c in _df_.columns:
        objs[c.replace('추세', '변화량') if c.endswith('추세') else c + '변화량'] = _df_[c].diff()
        objs[c.replace('추세', '모멘텀') if c.endswith('추세') else c + '모멘텀'] = _df_[c].diff().diff()
    return _df_.join(other=pd.concat(objs=objs, axis=1), how='left')

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

def calc_answer(data: pd.DataFrame, by :str ='종가', td :int =20, yld :float =5.0) -> pd.DataFrame:
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
    for i in range(len(calc)):
        if i > (len(calc) - td):
            continue
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
        5: [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        10: [-3, -2, -1, 1, 2, 3],
        15: [-4, -2.5, -1, 1, 2.5, 4],
        20: [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
        40: [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
    }
    for day, bound in thres.items():
        calc[f'{day}TD수익률'] = round(100 * calc[by].pct_change(periods=day).shift(-day).fillna(0), 2)
        cindex = [calc[f'{day}TD수익률'].min()] + bound + [calc[f'{day}TD수익률'].max()]
        calc[f'{day}TD색상'] = pd.cut(calc[f'{day}TD수익률'], bins=cindex, labels=scale, right=True)
        calc[f'{day}TD색상'].fillna(scale[0])
    return calc.drop(columns=['시가', '저가', '고가', '종가'])

def fetch_finance(ticker :str) -> (pd.DataFrame, pd.DataFrame):
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


meta_stock = stocks()
meta_index = indices(mode='raw')
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
    :: dropline: 낙폭
    :: finance: 재무제표 데이터
    """
    def __init__(self,
                 ticker :str,
                 on :str ='종가',
                 end_date :datetime =datetime.today(),
                 time_stamp :int =5,
                 mode :str ='offline',
                 filter_type :str ='lowpass'):
        """
        marketport @GITHUB 주가/지수 데이터 분석
        :param ticker: 종목코드/지수코드
        :param on: 가이드라인/모멘텀선/수익률곡선/낙폭 계산 시 참조 가격 (종가 기준)
        :param end_date: 마감 일자
        :param time_stamp: 시계열 Cut 기준 연수(year)
        :param mode: offline - 로컬 사용 / online - GITHUB 서버 사용
        :param filter_type: 필터 종류
        """
        self.ticker = ticker
        self.key = on
        self.e_date = end_date
        self.f_type = filter_type

        self.mkind = 'stock' if len(ticker) == 6 else 'index'
        meta = meta_stock if self.mkind == 'stock' else meta_index
        self.equity = meta.loc[ticker, '종목명']

        self.basis = pd.read_csv(
            f'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/series/{ticker}.csv',
            encoding='utf-8',
            index_col='날짜'
        ) if mode == 'online' else pd.read_csv(
            os.path.join(__root__, f'warehouse/series/{ticker}.csv'),
            encoding='utf-8',
            index_col='날짜'
        )
        self.basis.index = pd.to_datetime(self.basis.index)
        if not end_date.date() == datetime.today().date():
            self.basis = self.basis[self.basis.index <= end_date]
        if time_stamp:
            self.tic = self.basis.index[-1] - timedelta(365 * time_stamp)
            self.basis = self.basis[self.basis.index >= self.tic]

        self.g_line = pd.DataFrame()    # 필터선
        self.y_line = pd.DataFrame()    # 수익선
        self.t_line = pd.DataFrame()    # 추세선
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
            window_or_cutoff=[5, 10, 20, 60, 120],
            mode=self.f_type
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

        self.t_line = calc_trending(data=self.guideline)
        return self.t_line

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
    def show_price(self, filter_hover :bool =False) -> go.Figure:
        """
        단일 종목 가격/지수 그래프
        :return:
        """
        xtick = '주가' if self.mkind == 'stock' else '지수'
        unit = '원' if self.mkind == 'stock' else ''
        fig = go.Figure()

        src = self.guideline
        dform = ['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index]
        for col in src.columns:
            hover = col + '<br>필터: %{y:.2f}<br>날짜: %{customdata}' if filter_hover else f'{col}'
            fig.add_trace(
                go.Scatter(
                    x=src.index,
                    y=src[col],
                    name=col,
                    mode='lines',
                    line=dict(dash='dot' if 'MAF' in col else 'dash'),
                    showlegend=True,
                    customdata=dform,
                    visible=True if col in ['MAF120D', 'MAF60D' ,'MAF20D', 'LPF60D', 'LPF05D'] else 'legendonly',
                    hovertemplate=hover + '<extra></extra>'
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
                    hovertemplate='날짜:%{customdata}<br>' + col + ':%{y:,}' + f'{unit}<extra></extra>',
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
        unit = '원' if self.mkind == 'stock' else ''
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        src = self.trendline.copy()
        dform = ['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index]
        for col in src.columns:
            if '변화량' in col or '모멘텀' in col:
                continue
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
                hovertemplate='날짜:%{meta}<br>종가:%{y:,}' + f'{unit}<extra></extra>',
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

        cols = [col for col in self.trendline.columns if '변화량' in col or '모멘텀' in col]
        raw = self.trendline.copy()[cols].dropna()
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
                args=[{'visible' :visible}]
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

    def show_sales(self, kind :str ='annual') -> go.Figure:
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


class estimate(frame):
    """
    종목 추세 평가 모델
    """

    def m_basic(self, mode :str ='actual') -> pd.DataFrame:
        """
        기본 모델 ::: <평가지표:기본모델>@Google Drive
        :param: mode: 'actual' - Daily 모델 판단 모드
                      'tester-all' - 백테스트 전체 모드
                      'tester-specific' - 백테스트 단일 종목 모드
        :return:
        """
        frm = pd.concat(objs=[
            self.basis, self.trendline
        ], axis=1)
        frm['중기모멘텀'] = frm['중기모멘텀'].rolling(5).mean().fillna(0)
        frm['중장기모멘텀'] = frm['중장기모멘텀'].rolling(5).mean().fillna(0)

        if mode == 'actual':
            est = frm.iloc[-1].to_dict()
            if est['중기변화량'] > 0 and est['중장기변화량'] > 0 and est['중기모멘텀'] > 0 and est['중장기모멘텀'] > 0 and est['중기추세'] > 0 and est['중장기추세'] > 0:
                est['투자적합성'] = '적합'
            else:
                est['투자적합성'] = '부적합'
            return pd.DataFrame(data=est, index=[self.ticker])
        elif mode.startswith('test'):
            invest = []
            for i, date in enumerate(frm.index):
                data = frm.loc[date].to_dict()
                if data['중기변화량'] > 0 and data['중장기변화량'] > 0 and data['중기모멘텀'] > 0 and data['중장기모멘텀'] > 0 and data['중기추세'] > 0 and data['중장기추세'] > 0:
                    invest.append('적합' if mode.endswith('all') else data[self.key])
                else:
                    invest.append('부적합' if mode.endswith('all') else np.nan)
            frm['투자적합성'] = invest
            return frm




if __name__ == "__main__":
    print(stocks())
    # print(indices(mode='raw'))

    # asset = frame(ticker='005930', on='종가', time_stamp=5, mode='offline')
    # print(asset.equity)
    # print(asset.basis)
    # print(asset.guideline)
    # print(asset.yieldline)
    # print(asset.trendline)

    # display = vstock(ticker='044340', on='종가', end_date=datetime(2018, 10, 21), time_stamp=5, mode='offline', filter_type='butter')
    # display = vstock(ticker='044340', on='종가', end_date=datetime.today(), time_stamp=0, mode='offline')
    # display.show_price(filter_hover=True).show()
    # display.show_trend().show()
    # display.show_momentum().show()
    # display.show_drawdown().show()
    # display.show_sales(kind='annual').show()
    # display.show_sales(kind='quarter').show()
    # display.show_financial_ratio(kind='annual').show()
    # display.show_financial_ratio(kind='quarter').show()

    # model = estimate(ticker='204270', on='종가', time_stamp=5, mode='offline')
    # print(model.equity)
    # print(model.m_basic(mode='tester-specific'))

    ##################################################################################################################

    # from pykrx import stock
    # samples = []
    # for ind in ['1002', '1003', '2203']:
    # for ind in ['1003']:
    #     samples += stock.get_index_portfolio_deposit_file(ticker=ind)
    # samples = ['005930', '000660']
    #
    # report = []
    # for m, ticker in enumerate(samples):
    #     frm = frame(ticker=ticker, on='종가', end_date=datetime.today(), time_stamp=0, mode='offline')
    #
    #     price_line = frm.basis.drop(columns=['거래량'])
    #     index_date = price_line.index
    #     if len(price_line) < 252 * 2:
    #         continue
    #
    #     print(f'{100 * (m + 1) / len(samples)}%...{ticker} {frm.equity}  :: 성공률: ', end='')
    #     obj = {'종목명': frm.equity, '종목코드': ticker}
    #     if os.path.isfile(f'{ticker}.csv'):
    #         frm = pd.read_csv(f'{ticker}.csv', encoding='utf-8', index_col='날짜')
    #         frm.index = pd.to_datetime(frm.index)
    #     else:
    #         frm = pd.DataFrame()
    #         for n, date in enumerate(index_date[:-20]):
    #             if n <= 120:
    #                 continue
    #             _price_line = price_line[price_line.index <= date].copy()
    #             _guide_line = calc_filtered(_price_line.종가, window_or_cutoff=[5, 10, 20, 60, 120])
    #             _trend_line = calc_trending(data=_guide_line)
    #             _line = pd.concat([_price_line, _guide_line, _trend_line], axis=1)
    #             _line.index.name = '날짜'
    #             _line.reset_index(level=0, inplace=True)
    #
    #
    #             _line['중기변화량']
    #
    #             data = _line.iloc[-5].to_dict()
    #
    #             if data['중기변화량'] > 0 and data['중장기변화량'] > 0 and data['중기모멘텀'] > 0 and data['중장기모멘텀'] > 0:# and data['중기추세'] > 0 and data['중장기추세'] > 0:
    #                 data['투자적합성'] = '적합'
    #                 data['투자시기'] = data['종가']
    #             else:
    #                 data['투자적합성'] = '부적합'
    #                 data['투자시기'] = np.nan
    #
    #             answer_set = price_line[n+1:n+11].values.flatten()
    #             std = answer_set[0]
    #             data['투자성공'] = False
    #             if std == 0:
    #                 pass
    #             else:
    #                 for comp in answer_set[1:]:
    #                     if comp == 0:
    #                         continue
    #                     if (comp/std - 1) >= 0.03:
    #                         data['투자성공'] = True
    #                         break
    #             frm = frm.append(pd.DataFrame(data=data, index=[data['날짜']]))
    #         frm.to_csv(f'{ticker}.csv', encoding='utf-8', index=False)
    #
    #     m_recommend = frm[frm['투자적합성'] == '적합'].copy()
    #     if m_recommend.empty:
    #         continue
    #     m_success = m_recommend[m_recommend['투자성공'] == True].copy()
    #
    #     obj['성공률'] = 100 * len(m_success)/len(m_recommend)
    #     obj['시작일'] = frm.index[0].date()
    #     obj['종료일'] = frm.index[-1].date()
    #     obj['기간'] = (frm.index[-1] - frm.index[0]).days
    #     obj['적합률'] = 100 * len(m_recommend)/len(frm)
    #     report.append(obj)
    #     print(f'{obj["성공률"]:.2f}%')
    # df = pd.DataFrame(report)
    # df.to_csv(r'Report.csv', encoding='euc-kr', index=False)



    # ref_frm = frame(ticker=ticker, on='종가', end_date=datetime.today(), time_stamp=0, mode='offline')
    # dno_frm = pd.read_csv(rf'./{ticker}.csv', encoding='utf-8', index_col='날짜')
    # dno_frm.index = pd.to_datetime(dno_frm.index)
    # prc = ref_frm.basis
    # ref = ref_frm.guideline
    #
    # dno = dno_frm[[col for col in ref.columns if not '10' in col]].copy()
    #
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         x=prc.index,
    #         y=prc['종가'],
    #         name='종가'
    #     )
    # )
    # for col in ref.columns:
    #     if '10' in col:
    #         continue
    #     fig.add_trace(
    #         go.Scatter(
    #             x=ref.index,
    #             y=ref[col],
    #             name='평가'+col,
    #             visible='legendonly' if 'MAF' in col else True
    #         )
    #     )
    #
    # for col in dno.columns:
    #     if 'MAF' in col: continue
    #     fig.add_trace(
    #         go.Scatter(
    #             x=dno.index,
    #             y=dno[col],
    #             name='실제'+col,
    #             mode='lines',
    #             line=dict(dash='dot'),
    #         )
    #     )


    # fig = go.Figure()
    # src = dno_frm[[col for col in ref_frm.guideline.columns if not '10' in col]].copy()
    # for col in src.columns:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=src.index,
    #             y=src[col],
    #             name=col,
    #             mode='lines',
    #             line=dict(dash='dot' if 'MAF' in col else 'dash'),
    #             showlegend=True,
    #             visible=True if col in ['MAF120D', 'MAF60D', 'MAF20D', 'LPF60D', 'LPF05D'] else 'legendonly',
    #             hovertemplate=f'{col}<extra></extra>'
    #         )
    #     )
    #
    # src = dno_frm.copy()
    # dform = ['{}/{}/{}'.format(d.year, d.month, d.day) for d in src.index]
    # for col in ['종가', '시가', '저가', '고가']:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=src.index,
    #             y=src[col],
    #             customdata=dform,
    #             name=col,
    #             mode='lines',
    #             showlegend=True,
    #             visible=True if col == '종가' else 'legendonly',
    #             hovertemplate='날짜:%{customdata}<br>' + col + ':%{y:,}<extra></extra>',
    #         )
    #     )
    #
    # fig.add_trace(
    #     go.Candlestick(
    #         x=src.index,
    #         customdata=dform,
    #         open=src['시가'],
    #         high=src['고가'],
    #         low=src['저가'],
    #         close=src['종가'],
    #         increasing_line=dict(color='red'),
    #         decreasing_line=dict(color='blue'),
    #         name='일봉',
    #         visible='legendonly',
    #         showlegend=True,
    #     )
    # )

    # fig.update_layout(
    #     dict(
    #         title=f'<b>[{ref_frm.equity}({ticker})]</b> 가격 분석',
    #         plot_bgcolor='white',
    #         annotations=[
    #             dict(
    #                 text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
    #                 showarrow=False,
    #                 xref="paper", yref="paper",
    #                 x=0.005, y=-0.002
    #             )
    #         ],
    #         legend=dict(traceorder='reversed'),
    #         yaxis=dict(title=f'가격[KRW]', showgrid=True, zeroline=False, showticklabels=True, autorange=True,
    #                    gridcolor='lightgrey'),
    #         xaxis=dict(
    #             title='날짜', showgrid=True, zeroline=False, showticklabels=True, autorange=True, gridcolor='lightgrey',
    #             rangeselector=dict(
    #                 buttons=list([
    #                     dict(count=1, label="1m", step="month", stepmode="backward"),
    #                     dict(count=3, label="3m", step="month", stepmode="backward"),
    #                     dict(count=6, label="6m", step="month", stepmode="backward"),
    #                     dict(count=1, label="YTD", step="year", stepmode="todate"),
    #                     dict(count=1, label="1y", step="year", stepmode="backward"),
    #                     dict(count=3, label="3y", step="year", stepmode="backward"),
    #                     dict(step="all")
    #                 ])
    #             )
    #         ),
    #         xaxis_rangeslider=dict(visible=False)
    #     )
    # )
    # fig.update_traces(
    #     selector=dict(type='candlestick'),
    #     xhoverformat='%Y-%m-%d',
    #     yhoverformat=','
    # )
    # fig.show()


