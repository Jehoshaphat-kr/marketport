import os
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as of
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

root = os.path.join(
    os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop'),
    f'tdat/{datetime.today().strftime("%Y-%m-%d")}'
)
if not os.path.isdir(root):
    os.makedirs(name=root)


def reform(span):
    """
    날짜 형식 변경 (from)datetime --> (to)YY/MM/DD
    :param span: 날짜 리스트
    :return:
    """
    return [f'{d.year}/{d.month}/{d.day}' for d in span]

def obj_price(df:pd.DataFrame) -> dict:
    """
    기본 주가 차트 요소
    :param df: 주가 데이터프레임
    :return: dict() :: key = ['일봉', '시가', '고가', '저가', '종가']
    """
    require = ['시가', '고가', '저가', '종가']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for price data')

    objects = dict()
    objects['일봉'] = go.Candlestick(
        name='일봉', x=df.index,
        open=df['시가'], high=df['고가'], low=df['저가'], close=df['종가'],
        increasing_line=dict(color='red'), decreasing_line=dict(color='royalblue'),
        visible=True, showlegend=True,
    )

    for col in require:
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            line=dict(color='grey'),
            visible='legendonly', showlegend=True,
            meta=reform(span=df.index),
            hovertemplate=col + ': %{y:,}원<br>날짜: %{meta}<extra></extra>',
        )
    return objects

def obj_bollinger(df:pd.DataFrame, group:bool=False) -> dict:
    """
    볼린저밴드 차트 요소
    :param df: 볼린저밴드 데이터프레임
    :param group: 범례 그룹 여부
    :return: dict() :: key = ['상한선', '기준선', '하한선', '상한지시', '하한지시', '밴드폭', '신호']
    """
    require = ['상한선', '기준선', '하한선']
    columns = df.columns.values
    if not len([x for x in require if x in columns]) == len(require):
        raise KeyError(f'argument not sufficient for bollinger-band data')

    objects = dict()
    meta = reform(span=df.index)
    for n, col in enumerate(['상한선', '기준선', '하한선']):
        objects[col] = go.Scatter(
            name='볼린저밴드', x=df.index, y=df[col],
            mode='lines', line=dict(color='rgb(184, 247, 212)'), fill='tonexty' if n else None,
            visible='legendonly', showlegend=False if n else True,
            legendgroup='볼린저밴드',
            meta=meta,
            hovertemplate=col + '<br>날짜: %{meta}<br>값: %{y:,d}원<extra></extra>',
        )
    for n, col in enumerate(['상한지시', '하한지시']):
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            mode='markers', marker=dict(
                symbol=f'triangle-{"up" if n else "down"}', color='red' if n else 'royalblue', size=9
            ), visible='legendonly', showlegend=False,
            legendgroup='볼린저밴드',
            hoverinfo='skip'
        )
    for col in ['밴드폭', '신호']:
        objects[col] = go.Scatter(
            name=col, x=df.index, y=df[col],
            visible=True, showlegend=True,
            legendgroup='볼린저밴드' if group else None,
            meta=meta,
            hovertemplate=col + '<br>날짜: %{meta}<br>값: %{y:.2f}<extra></extra>'
        )
    return objects

def obj_pivot(df:pd.DataFrame) -> dict:
    """
    피벗 지점 차트 요소
    :param df: 피벗 지점 데이터프레임
    :return: dict() :: key = [고점, 저점]
    """
    objects = {}
    for n, col in enumerate(df.columns):
        sr = df[col].dropna()
        objects[col] = go.Scatter(
            name=f'피벗포인트', x=sr.index, y=sr,
            mode='markers', marker=dict(
                symbol='circle', color='blue' if col == '고점' else 'red', size=8, opacity=0.7
            ), visible='legendonly', showlegend=True if n else False,
            legendgroup='pivot',
            meta=reform(span=sr.index),
            hovertemplate='날짜: %{meta}<br>' + col + '피벗: %{y:,d}원<extra></extra>'
        )
    return objects

def obj_trend(df:pd.DataFrame) -> dict:
    """
    직선 추세 차트 요소
    :param df: 직선 추세 데이터프레임
    :return: dict() :: key = ['1Y평균저항선', '1Y평균지지선', '6M평균저항선', '6M평균지지선', '3M평균저항선', '3M평균지지선',
                              '1Y', '6M', '3M']
    """
    meta = reform(span=df.index)
    objects = {}
    for n, col in enumerate(df.columns):
        key = f"{col[:4] if '평균' in col else col[:2]}추세"
        objects[col] = go.Scatter(
            name=key, x=df.index, y=df[col],
            mode='lines', line=dict(color='royalblue' if col.endswith('저항선') else 'red'),
            visible='legendonly', showlegend=False if col.endswith('지지선') else True,
            legendgroup=key,
            meta=meta,
            hovertemplate='날짜: %{meta}<br>' + col + ': %{y:,d}원<extra></extra>'
        )
    return objects

class Chart:
    def __init__(self, obj):
        self.obj = obj
        return

    def layout_basic(self, title: str = '', x_title: str = '날짜', y_title: str = ''):
        """
        기본 차트 레이아웃
        :param title: 차트 제목
        :param x_title: x축 이름
        :param y_title: y축 이름
        :return:
        """
        return go.Layout(
            title=f'<b>{self.obj.name}[{self.obj.ticker}]</b> : {title}',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text="TDAT 내일모레, the-day-after-tomorrow.tistory.com",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            # legend=dict(traceorder='reversed'),
            yaxis=dict(
                title=f'{y_title}',
                showgrid=True, gridcolor='lightgrey',
                zeroline=False,
                showticklabels=True, autorange=True,
            ),
            xaxis=dict(
                title=f'{x_title}',
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

    def show_basic(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        주가 기본 분석 차트
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=2, cols=1, row_width=[0.15, 0.85], shared_xaxes=True, vertical_spacing=0.05)

        # 주가 차트
        for key, obj in obj_price(df=self.obj.price).items():
            fig.add_trace(obj, row=1, col=1)

        # 볼린저 밴드
        for key, obj in obj_bollinger(df=self.obj.bollinger).items():
            if key in ['상한선', '기준선', '하한선', '상한지시', '하한지시']:
                fig.add_trace(obj, row=1, col=1)

        # 피벗 포인트
        for key, obj in obj_pivot(df=self.obj.pivot).items():
            fig.add_trace(obj, row=1, col=1)

        # 평균 추세선
        for key, obj in obj_trend(df=self.obj.avg_trend).items():
            fig.add_trace(obj, row=1, col=1)

        # 필터선
        guide = self.obj.filters.copy()
        for col in guide.columns:
            if col.startswith('EMA') or col.startswith('IIR'):
                continue
            fig.add_trace(go.Scatter(
                x=guide.index,
                y=guide[col],
                name=col,
                legendgroup=col,
                visible='legendonly',
                meta=reform(span=guide.index),
                hovertemplate=col + '<br>값: %{y:.2f}<br>날짜: %{meta}<extra></extra>',
            ))

        # 거래량
        volume = self.obj.price['거래량']
        fig.add_trace(go.Bar(
            x=volume.index,
            y=volume.values,
            customdata=reform(span=self.obj.price.index),
            name='거래량',
            marker=dict(
                color=['blue' if d < 0 else 'red' for d in volume.pct_change().fillna(1)]
            ),
            showlegend=False,
            hovertemplate='날짜:%{customdata}<br>거래량:%{y:,}<extra></extra>'
        ), row=2, col=1)

        layout = self.layout_basic(title='기본 분석 차트', x_title='', y_title='가격(KRW)')
        layout.update(dict(
            xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='거래량', showgrid=True, gridcolor='lightgrey')
        ))
        fig.update_layout(layout)

        if show:
            fig.show()
        if save:
            of.plot(fig, filename=os.path.join(root, f"{self.obj.ticker}{self.obj.name}-기본차트.html"), auto_open=False)
        return fig

    def show_bollinger(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        볼린저 밴드
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=3, cols=1, row_width=[0.15, 0.15, 0.7], shared_xaxes=True, vertical_spacing=0.05)
        # 주가 차트
        for key, obj in obj_price(df=self.obj.price).items():
            fig.add_trace(obj, row=1, col=1)

        # 볼린저 밴드 차트
        for key, obj in obj_bollinger(df=self.obj.bollinger).items():
            if key in ['상한선', '기준선', '하한선', '상한지시', '하한지시']:
                fig.add_trace(obj, row=1, col=1)
            elif key == '밴드폭':
                fig.add_trace(obj, row=2, col=1)
            elif key == '신호':
                fig.add_trace(obj, row=3, col=1)
        fig.add_hline(y=1, row=3, col=1, line=dict(dash='dot', width=0.5, color='black'))
        fig.add_hline(y=0, row=3, col=1, line=dict(dash='dot', width=0.5, color='black'))


        # 레이아웃
        layout = self.layout_basic(title='볼린저 밴드 차트', x_title='', y_title='가격(KRW)')
        layout.update(dict(
            xaxis2=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='밴드폭', showgrid=True, gridcolor='lightgrey', zeroline=True),
            xaxis3=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis3=dict(title='매매구간', showgrid=True, gridcolor='lightgrey', zeroline=True)
        ))
        fig.update_layout(layout)
        if show:
            fig.show()
        if save:
            of.plot(fig, filename=os.path.join(root, f"{self.obj.ticker}{self.obj.name}-볼린저밴드.html"), auto_open=False)
        return fig

    def show_trend(self, show: bool = False, save: bool = False):
        """
        추세선 및 MACD
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=2, cols=1, row_width=[0.2, 0.8], shared_xaxes=True, vertical_spacing=0.05,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

        # 종가 정보
        price = self.obj.price['종가']
        tic = price.index[0]
        toc = price.index[-1]
        fig.add_trace(go.Scatter(
            x=price.index,
            y=price,
            meta=reform(span=price.index),
            name='종가',
            hovertemplate='날짜: %{meta}<br>종가: %{y:,}원<extra></extra>'
        ), row=1, col=1, secondary_y=False)

        # 추세선
        point = self.obj.bend_point.copy()
        trend = self.obj.guidance.copy()
        for col in trend.columns:
            if col.startswith('d'):
                continue
            fig.add_trace(go.Scatter(
                x=trend.index,
                y=trend[col],
                customdata=reform(span=trend.index),
                legendgroup=col,
                name=col,
                mode='lines',
                showlegend=True,
                visible='legendonly',
                hovertemplate=col + '<br>추세:%{y:.3f}<br>날짜:%{customdata}<br><extra></extra>',
            ), row=1, col=1, secondary_y=True)

            pick = point[f'det{col}'].dropna()
            fig.add_trace(go.Scatter(
                x=pick.index,
                y=pick['value'],
                mode='markers',
                text=pick['bs'],
                meta=reform(pick.index),
                legendgroup=col,
                showlegend=False,
                marker=dict(
                    color=pick['color'],
                    symbol=pick['symbol']
                ),
                visible='legendonly',
                hovertemplate='%{text}<br>날짜: %{meta}<br>값: %{y}<extra></extra>'
            ), row=1, col=1, secondary_y=True)

        # MACD
        data = self.obj.macd
        form = reform(span=data.index)
        for n, col in enumerate(['MACD', 'MACD-Sig']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                meta=form,
                legendgroup='macd',
                showlegend=True if not n else False,
                hovertemplate=col + '<br>날짜: %{meta}<extra></extra>'
            ), row=2, col=1, secondary_y=False)

        fig.add_trace(go.Bar(
            x=data.index,
            y=data['MACD-Hist'],
            meta=form,
            name='MACD-Hist',
            marker=dict(
                color=['blue' if v < 0 else 'red' for v in data['MACD-Hist'].values]
            ),
            showlegend=False,
            hovertemplate='날짜:%{meta}<br>히스토그램:%{y:.2f}<extra></extra>'
        ), row=2, col=1, secondary_y=True)

        pick = point['detMACD'].dropna()
        fig.add_trace(go.Scatter(
            x=pick.index,
            y=pick['value'],
            mode='markers',
            marker=dict(
                symbol=pick['symbol'],
                color=pick['color'],
            ),
            text=pick['bs'],
            meta=reform(span=pick.index),
            hovertemplate='%{text}<br>날짜: %{meta}<extra></extra>',
            legendgroup='macd',
            showlegend=False
        ), row=2, col=1)

        layout = self.layout_basic(title='추세 분석 차트', x_title='', y_title='종가[KRW]')
        layout.update(dict(
            xaxis=dict(range=[tic, toc]),
            xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='추세', showgrid=False, zeroline=True, zerolinecolor='grey', zerolinewidth=2),
            yaxis3=dict(title='MACD', showgrid=True, gridcolor='lightgrey')
        ))
        fig.update_layout(layout)

        if show:
            fig.show()
        if save:
            of.plot(fig, filename=os.path.join(root, f"{self.obj.ticker}{self.obj.name}-추세차트.html"), auto_open=False)
        return fig


# class Fundamental(fundamental):
#     def layout_basic(self):
#         return
#
#     def show_business(self, save: bool=False, show: bool=False):
#         """
#         기업 사업 개요 텍스트 저장 또는 출력
#         :param save:
#         :param show:
#         :return:
#         """
#         if save:
#             with open(os.path.join(root, f"{self.ticker}{self.name}-개요.txt"), 'w', encoding='utf-8') as file:
#                 file.write(self.summary)
#
#         if show:
#             print(self.summary)
#         return
#
#     def show_summary(self, save: bool = False, show: bool = False) -> go.Figure:
#         """
#         [0, 0] 매출 제품 비중
#         [0, 1] 멀티 팩터
#         [1, 0] 컨센서스
#         [1, 1] 외국인 지분율
#         :param save:
#         :param show:
#         :return:
#         """
#         fig = make_subplots(
#             rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.1,
#             subplot_titles=(" ", "컨센서스", "외국인 보유비중", "차입공매도 비중"),
#             specs=[[{"type": "polar"}, {"type": "xy"}],
#                    [{"type": "xy", "secondary_y": True}, {"type": "xy", 'secondary_y': True}]]
#         )
#         # 멀티 팩터
#         df = self.multi_factor
#         for n, col in enumerate(df.columns):
#             fig.add_trace(go.Scatterpolar(
#                 name=col,
#                 r=df[col].astype(float),
#                 theta=df.index,
#                 fill='toself',
#                 showlegend=True,
#                 visible='legendonly' if n else True,
#                 hovertemplate=col + '<br>팩터: %{theta}<br>값: %{r}<extra></extra>'
#             ), row=1, col=1)
#
#         # 컨센서스
#         df = self.consensus.copy()
#         for col in ['목표주가', '종가']:
#             sr = df[col].dropna()
#             fig.add_trace(go.Scatter(
#                 name=col,
#                 x=sr.index,
#                 y=sr.astype(int),
#                 meta=reform(df.index),
#                 hovertemplate='날짜: %{meta}<br>' + col + ': %{y:,}원<extra></extra>'
#             ), row=1, col=2)
#
#         # 외국인보유비중
#         df = self.foreigner.copy()
#         for col in df.columns:
#             flag_price = col.startswith('종가')
#             form = ': %{y:,}원' if flag_price else ': %{y}%'
#             fig.add_trace(go.Scatter(
#                 name=col + '(지분율)' if flag_price else col,
#                 x=df.index,
#                 y=df[col].astype(int if flag_price else float),
#                 meta=reform(df.index),
#                 hovertemplate='날짜: %{meta}<br>' + col + form + '<extra></extra>'
#             ), row=2, col=1, secondary_y=False if flag_price else True)
#
#         # 차입공매도비중
#         df = self.short_sell.copy()
#         for col in df.columns:
#             is_price = col.endswith('종가')
#             form = ': %{y:,}원' if is_price else ': %{y}%'
#             fig.add_trace(go.Scatter(
#                 name=col,
#                 x=df.index,
#                 y=df[col].astype(int if is_price else float),
#                 meta=reform(df.index),
#                 hovertemplate='날짜: %{meta}<br>' + col + form + '<extra></extra>'
#             ), row=2, col=2, secondary_y=False if is_price else True)
#
#         # 레이아웃
#         fig.update_layout(dict(
#             title=f'<b>{self.name}[{self.ticker}]</b> : 기업 평가 및 수급',
#             plot_bgcolor='white'
#         ))
#         fig.update_xaxes(title_text="날짜", showgrid=True, gridcolor='lightgrey')
#         fig.update_yaxes(title_text="주가[원]", showgrid=True, gridcolor='lightgrey', row=1, col=2)
#         fig.update_yaxes(title_text="주가[원]", showgrid=True, gridcolor='lightgrey', row=2, col=1, secondary_y=False)
#         fig.update_yaxes(title_text="비중[%]", showgrid=False, row=2, col=1, secondary_y=True)
#         fig.update_yaxes(title_text="주가[원]", showgrid=True, gridcolor='lightgrey', row=2, col=2, secondary_y=False)
#         fig.update_yaxes(title_text="비중[%]", showgrid=False, row=2, col=2, secondary_y=True)
#         if show:
#             fig.show()
#         if save:
#             of.plot(fig, filename=os.path.join(root, f"{self.ticker}{self.name}-수급평가.html"), auto_open=False)
#         return fig
#
#     def show_sales(self, save: bool = False, show: bool = False) -> go.Figure:
#         """
#         [0, 0] 연간 시가총액/매출/영업이익/당기순이익
#         [0, 1] 분기 시가총액/매출/영업이익/당기순이익
#         [1, 0] 판관비/매출원가/R&D투자비융
#         [1, 1] 자산/부채/자본
#         :param save:
#         :param show:
#         :return:
#         """
#         fig = make_subplots(rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.05,
#                             subplot_titles=("매출 비중", "연간 실적", "SG&A, 매출원가 및 R&D투자", "자산"),
#                             specs=[[{"type": "pie"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]])
#
#         df = self.sales_product.copy()
#         fig.add_trace(go.Pie(
#             name='Product',
#             labels=df.index,
#             values=df,
#             textinfo='label+percent',
#             insidetextorientation='radial',
#             showlegend=False,
#             hoverinfo='label+percent'
#         ), row=1, col=1)
#
#         df_a = self.annual_statement
#         key = '매출액'
#         key = '순영업수익' if '순영업수익' in df_a.columns else key
#         key = '보험료수익' if '보험료수익' in df_a.columns else key
#         for n, col in enumerate(['시가총액', key, '영업이익', '당기순이익']):
#             y = df_a[col].fillna(0).astype(int)
#             fig.add_trace(go.Bar(
#                 x=df_a.index,
#                 y=y,
#                 name=f'연간{col}',
#                 marker=dict(color=colors[n]),
#                 legendgroup=col,
#                 meta=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in y],
#                 hovertemplate=col + ': %{meta}억원<extra></extra>',
#                 opacity=0.9,
#             ), row=1, col=2)
#
#         summary = pd.concat(objs=[self.sg_a, self.sales_cost, self.rnd_invest['R&D투자비중']], axis=1)
#         summary.sort_index(inplace=True)
#         for n, col in enumerate(summary.columns):
#             fig.add_trace(go.Bar(
#                 x=summary.index,
#                 y=summary[col].astype(float),
#                 name=col,
#                 hovertemplate=col + ': %{y}%<extra></extra>',
#                 opacity=0.9,
#             ), row=2, col=1)
#
#         asset = df_a['자산총계'].fillna(0).astype(int)
#         debt = df_a['부채총계'].fillna(0).astype(int)
#         capital = df_a['자본총계'].fillna(0).astype(int)
#         fig.add_trace(go.Bar(
#             x=df_a.index,
#             y=asset,
#             name='자산',
#             text=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in asset],
#             meta=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in debt],
#             customdata=[str(_) if _ < 10000 else str(_)[:-4] + '조 ' + str(_)[-4:] for _ in capital],
#             hovertemplate='자산: %{text}억원<br>부채: %{meta}억원<br>자본: %{customdata}억원<extra></extra>',
#             texttemplate=' ',
#             marker=dict(color='green'),
#             offsetgroup=0,
#             opacity=0.9,
#             showlegend=False
#         ), row=2, col=2)
#
#         fig.add_trace(go.Bar(
#             x=df_a.index,
#             y=debt,
#             name='부채',
#             hoverinfo='skip',
#             marker=dict(color='red'),
#             offsetgroup=0,
#             opacity=0.8,
#             showlegend=False
#         ), row=2, col=2)
#
#         fig.update_layout(dict(
#             title=f'<b>{self.name}[{self.ticker}]</b> : 실적, 지출 및 자산',
#             plot_bgcolor='white'
#         ))
#         fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=1, col=1)
#         fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=1, col=2)
#         fig.update_yaxes(title_text="비율[%]", gridcolor='lightgrey', row=2, col=1)
#         fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=2, col=2)
#
#         if show:
#             fig.show()
#         if save:
#             of.plot(fig, filename=os.path.join(root, f"{self.ticker}{self.name}-실적.html"), auto_open=False)
#         return fig
#
#     def show_multiple(self, save: bool = False, show: bool = False) -> go.Figure:
#         """
#         [0, 0] 연간 재무비율:: ROE/ROA/영업이익률
#         [0, 1] 분기 재무비율:: ROE/ROA/영업이익률
#         [1, 0] 연간 투자배수:: PER/PBR/PSR/PEG
#         [1, 1] 배당 수익률
#
#         :param save:
#         :param show:
#         :return:
#         """
#         fig = make_subplots(rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.05,
#                             subplot_titles=("연간 재무비율", "분기 재무비율", "투자 배수", "EPS, BPS"))
#
#         df_a = self.annual_statement
#         df_q = self.quarter_statement
#         for n, col in enumerate(['ROA', 'ROE', '영업이익률']):
#             fig.add_trace(go.Bar(
#                 x=df_a.index,
#                 y=df_a[col],
#                 name=f'연간{col}',
#                 marker=dict(color=colors[n]),
#                 legendgroup=col,
#                 hovertemplate=col + ': %{y}%<extra></extra>',
#                 opacity=0.9,
#             ), row=1, col=1)
#
#             fig.add_trace(go.Bar(
#                 x=df_q.index,
#                 y=df_q[col],
#                 name=f'분기{col}',
#                 marker=dict(color=colors[n]),
#                 legendgroup=col,
#                 hovertemplate=col + ': %{y}%<extra></extra>',
#                 opacity=0.9,
#             ), row=1, col=2)
#
#         for n, col in enumerate(['PER', 'PBR', 'PSR', 'PEG']):
#             fig.add_trace(go.Bar(
#                 x=df_a.index,
#                 y=df_a[col],
#                 name=col,
#                 hovertemplate=col + ': %{y}<extra></extra>',
#                 opacity=0.9
#             ), row=2, col=1)
#
#         for n, col in enumerate(['EPS(원)', 'BPS(원)']):
#             fig.add_trace(go.Bar(
#                 x=df_a.index,
#                 y=df_a[col],
#                 name=col.replace("(원)", ""),
#                 hovertemplate=col + ': %{y:,}원<extra></extra>',
#                 opacity=0.9
#             ), row=2, col=2)
#
#         fig.update_layout(dict(
#             title=f'<b>{self.name}[{self.ticker}]</b> : 투자 비율 및 배수',
#             plot_bgcolor='white'
#         ))
#         fig.update_yaxes(title_text="%", gridcolor='lightgrey', row=1, col=1)
#         fig.update_yaxes(title_text="%", gridcolor='lightgrey', row=1, col=2)
#         fig.update_yaxes(title_text="-", gridcolor='lightgrey', row=2, col=1)
#         fig.update_yaxes(title_text="원", gridcolor='lightgrey', row=2, col=2)
#
#         if show:
#             fig.show()
#         if save:
#             of.plot(fig, filename=os.path.join(root, f"{self.ticker}{self.name}-배수비율.html"), auto_open=False)
#         return fig