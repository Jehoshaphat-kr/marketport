import os
import tdatool as tt
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as of
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


def save_as(fig:go.Figure, filename:str):
    """
    차트 파일 저장
    :param fig: 차트 요소 
    :param filename: 저장 파일명
    :return: 
    """
    root = os.path.join(
        os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop'),
        f'tdat/{datetime.today().strftime("%Y-%m-%d")}'
    )
    if not os.path.isdir(root):
        os.makedirs(name=root)
    of.plot(fig, filename=os.path.join(root, filename), auto_open=False)
    return

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
                    showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002
                )
            ],
            yaxis=dict(
                title=f'{y_title}',
                showgrid=True, gridcolor='lightgrey', showticklabels=True, zeroline=False, autorange=True,
            ),
            xaxis=dict(
                title=f'{x_title}',
                showgrid=True, gridcolor='lightgrey', showticklabels=True, zeroline=False, autorange=True,rangeselector=dict(
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
        for key, obj in tt.trace_price(df=self.obj.price):
            fig.add_trace(obj, row=1, col=1)

        # 볼린저 밴드
        for key, obj in tt.trace_bollinger(df=self.obj.bollinger):
            if key in ['상한선', '기준선', '하한선']:
                fig.add_trace(obj, row=1, col=1)

        # 피벗 포인트
        for key, obj in tt.trace_pivot(df=self.obj.pivot):
            fig.add_trace(obj, row=1, col=1)

        # 추세선
        for key, obj in tt.trace_trend(df=self.obj.trend):
            fig.add_trace(obj, row=1, col=1)

        # 필터선
        for key, obj in tt.trace_filters(df=self.obj.filters):
            if key.startswith('SMA'):
                fig.add_trace(obj, row=1, col=1)

        # 거래량
        for key, obj in tt.trace_volume(df=self.obj.price, is_main=False):
            fig.add_trace(obj, row=2, col=1)

        layout = self.layout_basic(title='기본 분석 차트', x_title='', y_title='가격(KRW)')
        layout.update(dict(
            xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='거래량', showgrid=True, gridcolor='lightgrey')
        ))
        fig.update_layout(layout)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.obj.ticker}{self.obj.name}-기본차트.html")
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
        for key, obj in tt.trace_price(df=self.obj.price):
            fig.add_trace(obj, row=1, col=1)

        # 볼린저 밴드 차트
        for key, obj in tt.trace_bollinger(df=self.obj.bollinger):
            if key in ['상한선', '기준선', '하한선', '상한지시', '하한지시']:
                fig.add_trace(obj, row=1, col=1)
            elif key == '신호':
                fig.add_trace(obj, row=2, col=1)
            elif key == '밴드폭':
                fig.add_trace(obj, row=3, col=1)
        fig.add_hline(y=1, row=2, col=1, line=dict(dash='dot', width=0.5, color='black'))
        fig.add_hline(y=0, row=2, col=1, line=dict(dash='dot', width=0.5, color='black'))

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
            save_as(fig=fig, filename=f"{self.obj.ticker}{self.obj.name}-볼린저밴드.html")
        return fig

    def show_rsi(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        상대 강도 차트
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=3, cols=1, row_width=[0.15, 0.15, 0.7], shared_xaxes=True, vertical_spacing=0.05)

        # 주가 차트
        for key, obj in tt.trace_price(df=self.obj.price):
            fig.add_trace(obj, row=1, col=1)

        # RSI 차트
        for key, obj in tt.trace_rsi(df=self.obj.rsi):
            if key == 'RSI':
                fig.add_trace(obj, row=2, col=1)
            elif key.startswith('STOCH'):
                fig.add_trace(obj, row=3, col=1)
            else:
                fig.add_trace(obj, row=3, col=1)
        fig.add_hline(y=70, row=2, col=1, line=dict(dash='dot', width=0.5, color='black'))
        fig.add_hline(y=30, row=2, col=1, line=dict(dash='dot', width=0.5, color='black'))

        # 레이아웃
        layout = self.layout_basic(title='RSI(상대 강세) 차트', x_title='', y_title='가격(KRW)')
        layout.update(dict(
            xaxis2=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='RSI', showgrid=True, gridcolor='lightgrey', zeroline=True),
            xaxis3=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis3=dict(title='STOCH-RSI', showgrid=True, gridcolor='lightgrey', zeroline=True)
        ))
        fig.update_layout(layout)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.obj.ticker}{self.obj.name}-RSI.html")
        return fig

    def show_overview(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        사업 개요
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.1,
            subplot_titles=("제품 구성", "자산", "연간 실적", "분기 실적"),
            specs=[[{"type": "pie"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )

        # 제품 구성
        for key, obj in tt.trace_product(df=self.obj.product):
            fig.add_trace(obj, row=1, col=1)

        # 자산 현황
        for key, obj in tt.trace_asset(df=self.obj.annual):
            fig.add_trace(obj, row=1, col=2)

        # 연간 실적
        for key, obj in tt.trace_sales(df=self.obj.annual):
            fig.add_trace(obj, row=2, col=1)

        # 분기 실적
        for key, obj in tt.trace_sales(df=self.obj.quarter):
            fig.add_trace(obj, row=2, col=2)

        fig.update_layout(dict(
            title=f'<b>{self.obj.name}[{self.obj.ticker}]</b> : 제품, 자산 및 실적',
            plot_bgcolor='white'
        ))
        fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=1, col=2)
        fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=2, col=1)
        fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=2, col=2)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.obj.ticker}{self.obj.name}-개요.html")
        return fig

    def show_supply(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        수급 현황
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.1,
            subplot_titles=("컨센서스", "외국인 보유비중", "차입공매도 비중", "대차잔고 비중"),
            specs=[[{"type": "xy"}, {"type": "xy", "secondary_y": True}],
                   [{"type": "xy", "secondary_y": True}, {"type": "xy", 'secondary_y': True}]]
        )

        # 컨센서스 차트
        for key, obj in tt.trace_consensus(df=self.obj.consensus):
            fig.add_trace(obj, row=1, col=1)

        # 외국인 보유비중 차트
        for key, obj in tt.trace_foreigners(df=self.obj.foreigner):
            secondary_y = True if key == '종가' else False
            fig.add_trace(obj, row=1, col=2, secondary_y=secondary_y)

        # 차입공매도 비중
        for key, obj in tt.trace_shorts(df=self.obj.short):
            secondary_y = True if key == '수정 종가' else False
            fig.add_trace(obj, row=2, col=1, secondary_y=secondary_y)

        # 대차잔고 비중
        for key, obj in tt.trace_balance(df=self.obj.balance):
            secondary_y = True if key == '수정 종가' else False
            fig.add_trace(obj, row=2, col=2, secondary_y=secondary_y)

        # 레이아웃
        fig.update_layout(dict(
            title=f'<b>{self.obj.name}[{self.obj.ticker}]</b> : 수급 현황',
            plot_bgcolor='white'
        ))
        fig.update_xaxes(title_text="날짜", showgrid=True, gridcolor='lightgrey')
        fig.update_yaxes(title_text="주가[원]", showgrid=True, gridcolor='lightgrey', row=1, col=1)
        for row, col in ((1, 2), (2, 1), (2, 2)):
            fig.update_yaxes(title_text="주가[원]", showgrid=True, gridcolor='lightgrey', row=row, col=col, secondary_y=True)
            fig.update_yaxes(title_text="비중[%]", showgrid=False, row=row, col=col, secondary_y=False)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.obj.ticker}{self.obj.name}-수급평가.html")
        return fig

    # def show_trend(self, show: bool = False, save: bool = False):
    #     """
    #     추세선 및 MACD
    #     :param show:
    #     :param save:
    #     :return:
    #     """
    #     fig = make_subplots(rows=2, cols=1, row_width=[0.2, 0.8], shared_xaxes=True, vertical_spacing=0.05,
    #                         specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    #
    #     # 종가 정보
    #     price = self.obj.price['종가']
    #     tic = price.index[0]
    #     toc = price.index[-1]
    #     fig.add_trace(go.Scatter(
    #         x=price.index,
    #         y=price,
    #         meta=reform(span=price.index),
    #         name='종가',
    #         hovertemplate='날짜: %{meta}<br>종가: %{y:,}원<extra></extra>'
    #     ), row=1, col=1, secondary_y=False)
    #
    #     # 추세선
    #     point = self.obj.bend_point.copy()
    #     trend = self.obj.guidance.copy()
    #     for col in trend.columns:
    #         if col.startswith('d'):
    #             continue
    #         fig.add_trace(go.Scatter(
    #             x=trend.index,
    #             y=trend[col],
    #             customdata=reform(span=trend.index),
    #             legendgroup=col,
    #             name=col,
    #             mode='lines',
    #             showlegend=True,
    #             visible='legendonly',
    #             hovertemplate=col + '<br>추세:%{y:.3f}<br>날짜:%{customdata}<br><extra></extra>',
    #         ), row=1, col=1, secondary_y=True)
    #
    #         pick = point[f'det{col}'].dropna()
    #         fig.add_trace(go.Scatter(
    #             x=pick.index,
    #             y=pick['value'],
    #             mode='markers',
    #             text=pick['bs'],
    #             meta=reform(pick.index),
    #             legendgroup=col,
    #             showlegend=False,
    #             marker=dict(
    #                 color=pick['color'],
    #                 symbol=pick['symbol']
    #             ),
    #             visible='legendonly',
    #             hovertemplate='%{text}<br>날짜: %{meta}<br>값: %{y}<extra></extra>'
    #         ), row=1, col=1, secondary_y=True)
    #
    #     # MACD
    #     data = self.obj.macd
    #     form = reform(span=data.index)
    #     for n, col in enumerate(['MACD', 'MACD-Sig']):
    #         fig.add_trace(go.Scatter(
    #             x=data.index,
    #             y=data[col],
    #             name=col,
    #             meta=form,
    #             legendgroup='macd',
    #             showlegend=True if not n else False,
    #             hovertemplate=col + '<br>날짜: %{meta}<extra></extra>'
    #         ), row=2, col=1, secondary_y=False)
    #
    #     fig.add_trace(go.Bar(
    #         x=data.index,
    #         y=data['MACD-Hist'],
    #         meta=form,
    #         name='MACD-Hist',
    #         marker=dict(
    #             color=['blue' if v < 0 else 'red' for v in data['MACD-Hist'].values]
    #         ),
    #         showlegend=False,
    #         hovertemplate='날짜:%{meta}<br>히스토그램:%{y:.2f}<extra></extra>'
    #     ), row=2, col=1, secondary_y=True)
    #
    #     pick = point['detMACD'].dropna()
    #     fig.add_trace(go.Scatter(
    #         x=pick.index,
    #         y=pick['value'],
    #         mode='markers',
    #         marker=dict(
    #             symbol=pick['symbol'],
    #             color=pick['color'],
    #         ),
    #         text=pick['bs'],
    #         meta=reform(span=pick.index),
    #         hovertemplate='%{text}<br>날짜: %{meta}<extra></extra>',
    #         legendgroup='macd',
    #         showlegend=False
    #     ), row=2, col=1)
    #
    #     layout = self.layout_basic(title='추세 분석 차트', x_title='', y_title='종가[KRW]')
    #     layout.update(dict(
    #         xaxis=dict(range=[tic, toc]),
    #         xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
    #         yaxis2=dict(title='추세', showgrid=False, zeroline=True, zerolinecolor='grey', zerolinewidth=2),
    #         yaxis3=dict(title='MACD', showgrid=True, gridcolor='lightgrey')
    #     ))
    #     fig.update_layout(layout)
    #
    #     if show:
    #         fig.show()
    #     if save:
    #         of.plot(fig, filename=os.path.join(root, f"{self.obj.ticker}{self.obj.name}-추세차트.html"), auto_open=False)
    #     return fig


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