import pandas as pd
from tdatool.visualize import display as stock


class estimate(stock):
    def __init__(self, ticker: str = '005930', src: str = 'github', period: int = 5, meta = None):
        super().__init__(ticker=ticker, src=src, period=period, meta=meta)

        # Usage Frames
        self._spectra_ = pd.DataFrame()
        return

    @staticmethod
    def est_bollinger(bb_block:pd.DataFrame) -> pd.DataFrame:
        """

        :param bb_block:
        :return:
        """
        jong, mid, high, low = bb_block.종가, bb_block.기준선, bb_block.상한선, bb_block.하한선
        is_rise = mid[-3] < mid[-2] < mid[-1] and (mid[-1] / mid[-5] - 1) > 0
        is_fall = mid[-3] > mid[-2] > mid[-1] and (mid[-1] / mid[-5] - 1) < 0
        w = (
            ((mid[-1] - mid[-3]) / 2) / ((mid[-1] - mid[-5]) / 4)
        ) if is_rise
        return

    def est_bollinger(self) -> pd.DataFrame:

        bb = pd.concat([self.price, self.bollinger], axis=1)

        data = []
        jong, mid, high, low = bb.종가, bb.기준선, bb.상한선, bb.하한선
        for i in range(4, len(bb)):
            is_rise = mid[i - 2] < mid[i - 1] < mid[i] and (mid[i] / mid[i-4] - 1) > 0
            is_fall = mid[i - 2] > mid[i - 1] > mid[i] and (mid[i] / mid[i-4] - 1) < 0

            w = ((mid[i] - mid[i - 2]) / 2) / ((mid[i] - mid[i - 4]) / 4)
            locate = 1 if jong[i] > mid[i] else 0
            h_reserve = 100 * (high[i] / jong[i] - 1)
            l_reserve = 100 * (mid[i] / jong[i] - 1)

            data.append(
                w * h_reserve if is_rise else (w ** -1) * l_reserve if is_fall else h_reserve if locate else l_reserve
            )
        score = pd.Series(data=data, index=bb.index[4:], name='점수')
        return bb.join(score, how='left')

    @property
    def spectra(self):
        """
        거래일 기간 수익률 이산 색상 데이터프레임
        :return:
        """
        if self._spectra_.empty:
            scale = ['#F63538', '#BF4045', '#8B444E', '#414554', '#35764E', '#2F9E4F', '#30CC5A']
            thres = {
                5: [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
                10: [-3, -2, -1, 1, 2, 3],
                15: [-4, -2.5, -1, 1, 2.5, 4],
                20: [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
                40: [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
            }
            price = self.price['종가']
            data = {}
            for day, bound in thres.items():
                earnings = round(100 * price.pct_change(periods=day).shift(-day).fillna(0), 2)
                data[f'{day}TD-Y'] = earnings
                bins = [earnings.min()] + bound + [earnings.max()]
                data[f'{day}TD-C'] = pd.cut(earnings, bins=bins, labels=scale, right=True).fillna(scale[0])
            self._spectra_ = pd.concat(objs=data, axis=1)
        return self._spectra_


if __name__ == "__main__":
    ev = estimate(ticker='006400', src='pykrx', period=10)

    ach = ev.historical_return
    est = ev.est_bollinger()
    frm = pd.concat([est, ach], axis=1)


    # scale = ['#F63538', '#BF4045', '#8B444E', '#414554', '#35764E', '#2F9E4F', '#30CC5A']
    key = 'max_return_in_20td'
    scale = ['#414554', '#35764E', '#2F9E4F', '#30CC5A']
    bins = [0.0, 1.0, 3.0, 5.0]
    frm['color'] = pd.cut(frm[key], bins=bins + [frm[key].max()], labels=scale, right=True)
    print(frm.columns)

    s_avg = frm['점수'].mean()
    s_std = frm['점수'].std()
    print(f"{ev.name}({ev.ticker})")
    print(f'key = {key}')
    print(f"전체 기간       : {len(frm)}")
    print(f"전체 평균스코어 : {s_avg}")
    print(f"       표준편차 : {s_std}")
    print(f"전체 평균수익률 : {round(frm[key].mean(), 2)} %")
    for s in [1, 2, 3]:
        lim = (s_avg + s * s_std)
        _frm = frm[frm['점수'] > lim].copy()
        avg = _frm[key].mean()
        n_ovr = len(_frm[_frm[key] > 5])
        print(f'스코어 {round(lim, 2)} 이상 비율        : {round(100 * len(_frm) / len(frm), 2)} %')
        print(f'스코어 {round(lim, 2)} 이상 평균 수익률 : {round(avg, 2)} %')
        print(f'스코어 {round(lim, 2)} 이상 달성률      : {round(100 * n_ovr/len(_frm), 2)} %')
        print('-' * 100)


    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frm['점수'], y=frm[key],
        mode='markers', marker=dict(color=frm.color),
        meta=[f'{d.year}/{d.month}/{d.day}' for d in frm.index],
        hovertemplate='날짜: %{meta}<br>점수: %{x}<br>수익률: %{y}%<extra></extra>'
    ))
    for s in [1, 2, 3]:
        fig.add_vline(x=s_avg + s * s_std, line=dict(dash='dot', color='black'))
    fig.update_xaxes(title='점수')
    fig.update_yaxes(title='max_return_in_10td')
    fig.show()