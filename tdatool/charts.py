import plotly.graph_objects as go
import plotly.offline as of
import pandas as pd
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


class chart:

    ticker = ''
    name = ''
    price = pd.DataFrame()
    guide = pd.DataFrame()
    trend = pd.DataFrame()
    bound = pd.DataFrame()
    limit = pd.DataFrame()
    macd = pd.DataFrame()
    macd_pick=pd.DataFrame()

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
        주가 기본 분석 차트
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=2, cols=1, row_width=[0.15, 0.85], shared_xaxes=True, vertical_spacing=0.05)

        # 가격차트::일봉
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

        # 가격차트::종가
        fig.add_trace(go.Scatter(
            x=price.index,
            y=price['종가'],
            name='종가',
            meta=self.format(span=price.index),
            line=dict(color='grey'),
            hovertemplate='종가: %{y}원<br>날짜: %{meta}<extra></extra>'
        ))

        # 추세선
        data = self.bound.copy()
        for col in data.columns:
            gap = [l for l in ['1Y', 'YTD', '6M', '3M'] if col.startswith(l)][0]
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                legendgroup=f'{gap}추세선',
                name=f'{gap}추세선',
                visible='legendonly',
                showlegend=True if col.endswith('UP') else False,
                hovertemplate=col + '<extra></extra>'
            ))

        # 지지/저항선
        support_resist = self.limit.copy()
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

        layout = self.layout_basic(title='기본 분석 차트', x_title='', y_title='가격(KRW)')
        layout.update(dict(
            xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='거래량', showgrid=True, gridcolor='lightgrey')
        ))
        fig.update_layout(layout)

        if show:
            fig.show()
        if save:
            of.plot(fig, filename="chart-basic.html", auto_open=False)
        return fig

    def s_trend(self, show:bool=False, save:bool=False):
        """
        추세선 및 MACD
        :param show:
        :param save:
        :return:
        """
        fig = make_subplots(rows=2, cols=1, row_width=[0.15, 0.85], shared_xaxes=True, vertical_spacing=0.05,
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

        # 종가 정보
        price = self.price['종가']
        fig.add_trace(go.Scatter(
            x=price.index,
            y=price,
            meta=self.format(span=price.index),
            name='종가',
            hovertemplate='날짜: %{meta}<br>종가: %{y:,}원<extra></extra>'
        ), row=1, col=1, secondary_y=False)

        # 추세선
        trend = self.trend.copy()
        for col in trend.columns:
            if col.startswith('d'):
                continue
            fig.add_trace(go.Scatter(
                x=trend.index,
                y=trend[col],
                customdata=self.format(span=trend.index),
                name=col,
                mode='lines',
                showlegend=True,
                visible=True if col.endswith('IIR') else 'legendonly',
                hovertemplate=col + '<br>추세:%{y:.3f}<br>날짜:%{customdata}<br><extra></extra>',
            ), row=1, col=1, secondary_y=True)

        # MACD
        form = self.format(span=self.macd.index)
        for col in ['MACD', 'signal']:
            fig.add_trace(go.Scatter(
                x=self.macd.index,
                y=self.macd[col],
                name='MACD-Sig' if col == 'signal' else col,
                meta=form,
                showlegend=True,
                hovertemplate=col+'<br>날짜: %{meta}<extra></extra>'
            ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=self.macd.index,
            y=self.macd['hist'],
            meta=form,
            name='MACD-Hist',
            marker=dict(
                color=['blue' if v < 0 else 'red' for v in self.macd['hist'].values]
            ),
            showlegend=False,
            hovertemplate='날짜:%{customdata}<br>히스토그램:%{y:.2f}<extra></extra>'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=self.macd_pick.index,
            y=self.macd_pick['value'],
            name='MACD B/S',
            mode='markers',
            marker=dict(
                symbol=self.macd_pick['symbol'],
                color=self.macd_pick['color'],
                size=7
            ),
            text=self.macd_pick['B/S'],
            meta=self.format(span=self.macd_pick.index),
            opacity=0.7,
            hovertemplate='%{text}<br>날짜: %{meta}<extra></extra>',
            visible='legendonly',
            showlegend=True
        ), row=2, col=1)

        layout = self.layout_basic(title='추세 분석 차트', x_title='', y_title='종가[KRW]')
        layout.update(dict(
            xaxis2=dict(title='날짜', showgrid=True, gridcolor='lightgrey'),
            yaxis2=dict(title='추세', showgrid=False, zeroline=True, zerolinecolor='grey', zerolinewidth=2),
            yaxis3=dict(title='MACD', showgrid=True, gridcolor='lightgrey')
        ))
        fig.update_layout(layout)

        if show:
            fig.show()
        if save:
            of.plot(fig, filename="chart-tendency.html", auto_open=False)
        return fig