import pandas as pd
from tdatool.visualize import display as stock


class estimate(stock):
    def __init__(self, ticker: str = '005930', src: str = 'github', period: int = 5, meta = None):
        super().__init__(ticker=ticker, src=src, period=period, meta=meta)

        self._ack_perform_ = []

        # Usage Frames
        self._achieve_ = pd.DataFrame(index=self.price.index)
        self._performance_ = pd.DataFrame()
        self._spectra_ = pd.DataFrame()
        return

    def achieve(self, trade_days:int=20, target_yield:float=5.0) -> pd.DataFrame:
        """
        주가 흐름(정답지)별 수익률 달성 여부
        :return:
        """
        key = f'{round(target_yield, 1)}Y_{trade_days}TD'
        if not key in self._ack_perform_:
            calc = self.price[['시가', '저가', '고가', '종가']].copy()
            data = {f'ACH_{key}': [False] * len(calc), f'LEN_{key}': [-1] * len(calc)}
            for i, date in enumerate(calc.index[:-(trade_days + 1)]):
                p_span = calc[i + 1: i + (trade_days + 1)].values.flatten()
                if p_span[0] == 0: continue

                for n, p in enumerate(p_span):
                    if 100 * (p / p_span[0] - 1) >= target_yield:
                        data[f'ACH_{key}'][i] = True
                        data[f'LEN_{key}'][i] = n // 4
                        break
            self._achieve_ = self._achieve_.join(pd.DataFrame(data=data, index=calc.index), how='left')
            self._ack_perform_.append(key)
        return self._achieve_[[f'ACH_{key}', f'LEN_{key}']].copy()

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

    @property
    def trend_strength(self) -> pd.DataFrame:
        """
        추세선 강도
        :return:
        """
        cols = self.trend.columns
        data = {}
        for col in cols:
            _part = self.trend[col].dropna()
            data[col[:-1]] = 100 * round(_part[-1]/_part[0]-1, 6)
        return pd.DataFrame(data=data, index=[self.ticker])

    @property
    def bollinger_width(self):
        """
        볼린저 밴드 10거래일 평균 폭 대비 최근 폭 오차
        :return:
        """
        width = self.bollinger['밴드폭'].values
        return 100 * (width[-1]/width[-10:].mean() - 1)

    @property
    def bollinger_low(self):
        """
        볼린저 밴드 5거래일 하한선 대비 일봉 오차
        :return:
        """
        df = self.price.join(self.bollinger, how='left')
        calc = (df['종가'] - df['하한선']).values
        return 100 * (calc[-1] / calc[-5:].mean() - 1)

if __name__ == "__main__":
    ev = estimate(ticker='006400', src='local')

    print(ev.achieve())
    # print(ev.spectra)
    # print(ev.trend_strength)