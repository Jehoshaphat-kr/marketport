import os
import tdatool.visualize.traces as tt
import plotly.graph_objects as go
import plotly.offline as of
from plotly.subplots import make_subplots
from datetime import datetime
from tdatool.frame import prices as stock


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

class display(stock):
    def __init__(self, ticker: str = '005930', src: str = 'github', period: int = 5, meta = None):
        super().__init__(ticker=ticker, src=src, period=period, meta=meta)
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
            title=f'<b>{self.name}[{self.ticker}]</b> : {title}',
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
        for key, obj in tt.trace_price(df=self.price):
            fig.add_trace(obj, row=1, col=1)
        # 볼린저 밴드
        for key, obj in tt.trace_bollinger(df=self.bollinger):
            if key in ['상한선', '기준선', '하한선']:
                fig.add_trace(obj, row=1, col=1)
        # 피벗 포인트
        for key, obj in tt.trace_pivot(df=self.pivot):
            fig.add_trace(obj, row=1, col=1)
        # 추세선
        for key, obj in tt.trace_trend(df=self.trend):
            fig.add_trace(obj, row=1, col=1)
        # 필터선
        for key, obj in tt.trace_filters(df=self.filters):
            if key.startswith('SMA'):
                fig.add_trace(obj, row=1, col=1)
        # 거래량
        for key, obj in tt.trace_volume(df=self.price, is_main=False):
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
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-기본차트.html")
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
        for key, obj in tt.trace_price(df=self.price):
            fig.add_trace(obj, row=1, col=1)

        # 볼린저 밴드 차트
        for key, obj in tt.trace_bollinger(df=self.bollinger):
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
            yaxis2=dict(title='매매구간', showgrid=True, gridcolor='lightgrey', zeroline=True),
            xaxis3=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis3=dict(title='밴드폭', showgrid=True, gridcolor='lightgrey', zeroline=True)
        ))
        fig.update_layout(layout)
        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-볼린저밴드.html")
        return fig

    def show_momentum(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        MACD / RSI / STC
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(
            rows=4, cols=1, row_width=[0.2, 0.2, 0.2, 0.4], shared_xaxes=True, vertical_spacing=0.04,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # 주가 차트
        for key, obj in tt.trace_price(df=self.price):
            fig.add_trace(obj, row=1, col=1)
        # MACD
        for key, obj in tt.trace_macd(df=self.macd):
            secondary_y = True if key.lower().endswith('hist') else False
            fig.add_trace(obj, row=2, col=1, secondary_y=secondary_y)
        # TRIX 차트
        for key, obj in tt.trace_trix(df=self.trix):
            fig.add_trace(obj, row=3, col=1)
        # STC
        for key, obj in tt.trace_stc(df=self.stc):
            fig.add_trace(obj, row=4, col=1)

        # 레이아웃
        layout = self.layout_basic(title='모멘텀', x_title='', y_title='가격(KRW)')
        fig.update_layout(layout)
        for label, row, col in (('MACD', 2, 1), ('TRIX', 3, 1), ('STC', 4, 1)):
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showticklabels=True, row=row, col=col)
            fig.update_yaxes(title_text=label, showgrid=True, gridcolor='lightgrey', showticklabels=True,
                             row=row, col=col, secondary_y=False)
        fig.update_yaxes(showticklabels=True, row=2, col=1, secondary_y=True)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-볼린저밴드.html")
        return fig

    def show_overtrade(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        상대 강도 차트
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=4, cols=1, row_width=[0.2, 0.2, 0.2, 0.4], shared_xaxes=True, vertical_spacing=0.04)

        # 주가 차트
        for key, obj in tt.trace_price(df=self.price):
            fig.add_trace(obj, row=1, col=1)
        # RSI
        for key, obj in tt.trace_rsi(df=self.rsi):
            fig.add_trace(obj, row=2, col=1)
        # STOCHASTIC-RSI 차트
        for key, obj in tt.trace_stoch_rsi(df=self.stoch_rsi):
            fig.add_trace(obj, row=3, col=1)
        # CCI 차트
        for key, obj in tt.trace_cci(df=self.cci):
            fig.add_trace(obj, row=4, col=1)

        # 레이아웃
        layout = self.layout_basic(title='과매수/매도 지표', x_title='', y_title='가격(KRW)')
        fig.update_layout(layout)
        for label, row, col in (('RSI', 2, 1), ('Stochastic RSI', 3, 1), ('CCI', 4, 1)):
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showticklabels=True, row=row, col=col)
            fig.update_yaxes(title_text=label, showgrid=True, gridcolor='lightgrey', showticklabels=True,
                             row=row, col=col, secondary_y=False)
        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-RSI.html")
        return fig

    def show_vortex(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        VORTEX 차트
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=3, cols=1, row_width=[0.2, 0.2, 0.6], shared_xaxes=True, vertical_spacing=0.04)

        # 주가 차트
        for key, obj in tt.trace_price(df=self.price):
            fig.add_trace(obj, row=1, col=1)
        # Vortex
        for key, obj in tt.trace_vortex(df=self.vortex):
            row = 3 if key.endswith('Diff') else 2
            fig.add_trace(obj, row=row, col=1)

        # 레이아웃
        layout = self.layout_basic(title='Vortex 지표', x_title='', y_title='가격(KRW)')
        fig.update_layout(layout)
        for label, row, col in (('Vortex', 2, 1), ('Vortex-Diff', 3, 1)):
            fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showticklabels=True, row=row, col=col)
            fig.update_yaxes(title_text=label, showgrid=True, gridcolor='lightgrey', showticklabels=True,
                             row=row, col=col, secondary_y=False)
        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-RSI.html")
        return fig

    def show_overview(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        사업 개요
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.1,
            subplot_titles=("제품 구성", "자산", "연간 실적", "분기 실적"),
            specs=[[{"type": "pie"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]]
        )

        # 제품 구성
        for key, obj in tt.trace_product(df=self.product):
            fig.add_trace(obj, row=1, col=1)
        # 자산 현황
        for key, obj in tt.trace_asset(df=self.annual):
            fig.add_trace(obj, row=1, col=2)
        # 연간 실적
        for key, obj in tt.trace_sales(df=self.annual):
            fig.add_trace(obj, row=2, col=1)
        # 분기 실적
        for key, obj in tt.trace_sales(df=self.quarter, is_annual=False):
            fig.add_trace(obj, row=2, col=2)

        fig.update_layout(dict(
            title=f'<b>{self.name}[{self.ticker}]</b> : 제품, 자산 및 실적',
            plot_bgcolor='white',
            margin=dict(l=0)
        ))
        fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=1, col=2)
        fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=2, col=1)
        fig.update_yaxes(title_text="억원", gridcolor='lightgrey', row=2, col=2)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-개요.html")
        return fig

    def show_relative(self, show: bool=False, save: bool=False) -> go.Figure:
        """
        상대지표(By Company Guide)
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.1,
            subplot_titles=("", "상대 수익률", "PER 비교", "EV/EBITA 비교"),
            specs=[[{"type": "polar"}, {"type": "xy"}], [{"type": "bar"}, {"type": "bar"}]]
        )

        # 멀티팩터
        for key, obj in tt.trace_factors(df=self.factors):
            fig.add_trace(obj, row=1, col=1)
        # 상대수익률
        for key, obj in tt.trace_relyield(df=self.relyield):
            fig.add_trace(obj, row=1, col=2)
        # 상대배수
        for key, obj in tt.trace_relmultiple(df=self.relmultiple):
            col = 1 if key.endswith('(PER)') else 2
            fig.add_trace(obj, row=2, col=col)

        # 레이아웃
        fig.update_layout(dict(
            title=f'<b>{self.name}[{self.ticker}]</b> : 시장 상대 지표',
            plot_bgcolor='white',
        ))
        fig.update_yaxes(title_text="%", gridcolor='lightgrey', row=1, col=2)
        fig.update_xaxes(gridcolor='lightgrey', row=1, col=2)
        fig.update_yaxes(title_text="PER[-]", gridcolor='lightgrey', row=2, col=1)
        fig.update_xaxes(title_text="년도", row=2, col=1)
        fig.update_yaxes(title_text="EV/EBITA[-]", gridcolor='lightgrey', row=2, col=2)
        fig.update_xaxes(title_text="년도", row=2, col=2)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-상대지표.html")
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
        for key, obj in tt.trace_consensus(df=self.consensus):
            fig.add_trace(obj, row=1, col=1)
        # 외국인 보유비중 차트
        for key, obj in tt.trace_foreigners(df=self.foreigner):
            secondary_y = True if key == '종가' else False
            fig.add_trace(obj, row=1, col=2, secondary_y=secondary_y)
        # 차입공매도 비중
        for key, obj in tt.trace_shorts(df=self.short):
            secondary_y = True if key == '수정 종가' else False
            fig.add_trace(obj, row=2, col=1, secondary_y=secondary_y)
        # 대차잔고 비중
        for key, obj in tt.trace_balance(df=self.balance):
            secondary_y = True if key == '수정 종가' else False
            fig.add_trace(obj, row=2, col=2, secondary_y=secondary_y)

        # 레이아웃
        fig.update_layout(dict(
            title=f'<b>{self.name}[{self.ticker}]</b> : 수급 현황',
            plot_bgcolor='white'
        ))
        fig.update_yaxes(title_text="주가[원]", showgrid=True, gridcolor='lightgrey', row=1, col=1)
        for row, col in ((1, 2), (2, 1), (2, 2)):
            fig.update_yaxes(title_text="주가[원]", showgrid=True, gridcolor='lightgrey', row=row, col=col, secondary_y=True)
            fig.update_yaxes(title_text="비중[%]", showgrid=False, row=row, col=col, secondary_y=False)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-수급평가.html")
        return fig

    def show_multiples(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        투자 배수
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.1,
            subplot_titles=("PSR", "PER", "PBR", "배당 수익률"),
            specs=[[{"type": "xy", "secondary_y": True}, {"type": "xy", "secondary_y": True}],
                   [{"type": "xy", "secondary_y": True}, {"type": "xy", 'secondary_y': True}]]
        )

        # PSR, 매출액
        for key, obj in tt.trace_multiple(df=self.multiples, require=['매출액', 'PSR']):
            secondary_y = True if key == 'PSR' else False
            fig.add_trace(obj, row=1, col=1, secondary_y=secondary_y)
        # PER, EPS
        for key, obj in tt.trace_multiple(df=self.multiples, require=['EPS', 'PER']):
            secondary_y = True if key == 'PER' else False
            fig.add_trace(obj, row=1, col=2, secondary_y=secondary_y)
        # PBR, BPS
        for key, obj in tt.trace_multiple(df=self.multiples, require=['BPS', 'PBR']):
            secondary_y = True if key == 'PBR' else False
            fig.add_trace(obj, row=2, col=1, secondary_y=secondary_y)
        # DIV, DPS
        for key, obj in tt.trace_multiple(df=self.multiples, require=['DPS', 'DIV']):
            secondary_y = True if key == 'DIV' else False
            fig.add_trace(obj, row=2, col=2, secondary_y=secondary_y)

        # 레이아웃
        fig.update_layout(dict(
            title=f'<b>{self.name}[{self.ticker}]</b> : 투자 배수',
            plot_bgcolor='white'
        ))
        for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
            fig.update_yaxes(title_text="KRW", showgrid=True, gridcolor='lightgrey', row=row, col=col, secondary_y=False)
            fig.update_yaxes(title_text="배수[-, %]", showgrid=False, row=row, col=col, secondary_y=True)
        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-투자배수.html")
        return fig

    def show_cost(self, show: bool = False, save: bool = False) -> go.Figure:
        """
        지출 비용 
       :param show:
        :param save:
        :return:
        """
        fig = make_subplots(
            rows=2, cols=2, vertical_spacing=0.11, horizontal_spacing=0.1,
            subplot_titles=("매출 원가", "판관비", "R&D투자 비중", "부채율"),
            specs=[[{"type": "xy", "secondary_y": True}, {"type": "xy", "secondary_y": True}],
                   [{"type": "xy", "secondary_y": True}, {"type": "xy", 'secondary_y': True}]]
        )

        # 매출원가
        for key, obj in tt.trace_cost(df=self.cost):
            fig.add_trace(obj, row=1, col=1)
        # 판관비
        for key, obj in tt.trace_sga(df=self.sgna):
            fig.add_trace(obj, row=1, col=2)
        # R&D 투자비중
        for key, obj in tt.trace_rnd(df=self.rnd):
            fig.add_trace(obj, row=2, col=1)
        # 부채비율
        for key, obj in tt.trace_debt(df=self.annual):
            fig.add_trace(obj, row=2, col=2)

        # 레이아웃
        fig.update_layout(dict(title=f'<b>{self.name}[{self.ticker}]</b> : 비용과 부채', plot_bgcolor='white'))
        for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
            fig.update_yaxes(title_text="비율[%]", showgrid=True, gridcolor='lightgrey', row=row, col=col)

        if show:
            fig.show()
        if save:
            save_as(fig=fig, filename=f"{self.ticker}{self.name}-지출비용.html")
        return fig

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


if __name__ == "__main__":
    api = display(ticker='006400', src='pykrx', period=10)
    # api.show_overview(show=False, save=True)
    # api.show_supply(show=False, save=True)
    # api.show_relative(show=True, save=False)
    # api.show_multiples(show=False, save=True)
    # api.show_cost(show=False, save=True)

    # api.show_basic(show=False, save=True)
    api.show_bollinger(show=True, save=True)
    # api.show_momentum(show=True, save=False)
    # api.show_overtrade(show=True, save=False)
    # api.show_vortex(show=True, save=False)