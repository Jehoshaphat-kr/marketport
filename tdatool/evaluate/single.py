import pandas as pd
from tdatool.visualize import display as stock


class estimate(stock):
    def __init__(self, ticker: str = '005930', src: str = 'github', period: int = 5, meta = None):
        super().__init__(ticker=ticker, src=src, period=period, meta=meta)

        # Usage Frames
        self._spectra_ = pd.DataFrame()
        return

    def est_bollinger(self):
        ach = self.historical_return.copy()
        bol = self.bollinger.copy()

        is_rising = lambda x: (x[-5] < x[-3] < x[-1]) and (x[-3] < x[-2] < x[-1])
        trd = bol.기준선.rolling(window=5).apply(is_rising)
        df = pd.concat([self.price, ach, bol, trd], axis=1)
        df.to_csv(r'./test.csv', encoding='euc-kr')
        return

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
    def trend_strength(self) -> dict:
        """
        추세선 강도
        :return:
        """
        cols = self.trend.columns
        data = {}
        for col in cols:
            _part = self.trend[col].dropna()
            data[col[:-1]] = 100 * round(_part[-1]/_part[0]-1, 6)
        return data

    @property
    def bollinger_width(self) -> dict:
        """
        볼린저 밴드 10거래일 평균 폭 대비 최근 폭 오차
        :return:
        """
        width = self.bollinger['밴드폭'].values
        return {'볼린저폭': 100 * (width[-1]/width[-10:].mean() - 1)}

    @property
    def bollinger_height(self) -> dict:
        """
        볼린저 밴드 5거래일 하한선 대비 일봉 오차
        :return:
        """
        df = self.price.join(self.bollinger, how='left')
        calc = (df['종가'] - df['하한선']).values
        return {'볼린저높이': 100 * (calc[-1] / calc[-5:].mean() - 1)}

if __name__ == "__main__":
    ev = estimate(ticker='006400', src='pykrx')

    print(ev.est_bollinger())
    # print(ev.spectra)
    # print(ev.trend_strength)