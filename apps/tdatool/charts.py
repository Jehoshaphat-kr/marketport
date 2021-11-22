import plotly.graph_objects as go
import plotly.offline as of
import pandas as pd
from plotly.subplots import make_subplots
from datum import analyze


class chart(analyze):
    @staticmethod
    def format(span):
        """
        날짜 형식 변경 (from)datetime --> (to)YY/MM/DD
        :param span: 날짜 리스트
        :return:
        """
        return [f'{d.year}/{d.month}/{d.day}' for d in span]

    def layout_basic(self, title:str='', x_title:str='날짜', y_title:str=''):
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

    def s_price(self, show:bool=False, save:bool=False) -> go.Figure:
        """

        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=2, cols=1, row_width=[0.15, 0.85], shared_xaxes=True, vertical_spacing=0.05)

        # 가격차트::종가/일봉
        price = self.price[['시가', '고가', '저가', '종가']].copy()
        fig.add_trace(go.Candlestick(
            x=price.index,
            open=price['시가'],
            high=price['고가'],
            low=price['저가'],
            close=price['종가'],
            increasing_line=dict(color='red'),
            decreasing_line=dict(color='blue'),
            name='일봉',
            visible='legendonly',
            showlegend=True,
        ))

        fig.add_trace(go.Scatter(
            x=price.index,
            y=price['종가'],
            name='종가',
            meta=self.format(span=price.index),
            line=dict(color='grey'),
            hovertemplate='종가: %{y}원<br>날짜: %{meta}<extra></extra>'
        ))

        # 거래량
        volume = self.price['거래량']
        fig.add_trace(go.Bar(
            x=volume.index,
            y=volume.values,
            customdata=self.format(span=self.price.index),
            name='거래량',
            marker=dict(
                color=['blue' if self.price.loc[d, '시가'] > self.price.loc[d, '종가'] else 'red' for d in volume.index]
            ),
            showlegend=False,
            hovertemplate='날짜:%{customdata}<br>거래량:%{y:,}<extra></extra>'
        ), row=2, col=1)

        # 필터선
        guide = self.guide.copy()
        for col in guide.columns:
            cond = col[-3:] == '60D' and (col.startswith('SMA') or col.startswith('IIR'))
            fig.add_trace(go.Scatter(
                x=guide.index,
                y=guide[col],
                name=col,
                visible=True if cond else 'legendonly',
                meta=self.format(span=guide.index),
                hovertemplate=col + ': %{y:,.2f}<br>날짜: %{meta}<extra></extra>'
            ))

        layout = self.layout_basic(title='기본 분석 차트', x_title='', y_title='가격(KRW)')
        layout.update(dict(
            # legend=dict(traceorder='normal'),
            # xaxis=dict(title='', showticklabels=True),
            xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='거래량', showgrid=True, gridcolor='lightgrey')
        ))
        fig.update_layout(layout)

        if show:
            fig.show()
        if save:
            of.plot(fig, filename="chart-basic.html", auto_open=False)
        return fig