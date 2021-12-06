from tdatool.visualize import Technical as tech
from tdatool.visualize import Fundamental as fund
import pandas as pd


class evaluation(tech, fund):

    def __init__(self, ticker: str, src: str = 'git', period:int = 5):
        tech.__init__(self, ticker=ticker, src=src, period=period)
        fund.__init__(self, ticker=ticker)

        self.ticker = ticker

        # Usage Frames
        self._performance_ = pd.DataFrame()
        self._spectra_ = pd.DataFrame()
        return

    @property
    def performance(self) -> pd.DataFrame:
        """
        주가 흐름(정답지)별 수익률 달성 여부
        :return:
        """
        if self._performance_.empty:
            calc = self.price[['시가', '저가', '고가', '종가']].copy()

            data = {'ACH': [False] * len(calc), 'DUE': [-1] * len(calc)}
            for i, date in enumerate(calc.index[:-21]):
                p_span = calc[i+1:i+21].values.flatten()
                if p_span[0] == 0: continue

                for n, p in enumerate(p_span):
                    if 100 * (p/p_span[0] - 1) >= 5.0:
                        data['ACH'][i] = True
                        data['DUE'][i] = n // 4
                        break
            self._performance_ = pd.DataFrame(data=data, index=calc.index)
        return self._performance_

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
    ev = evaluation(ticker='006400')
    # print(ev.performance)
    # print(ev.spectra)