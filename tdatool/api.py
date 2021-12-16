import pandas as pd
from tdatool.evaluate.single import estimate as stock

class analytic(stock):
    def __init__(self, ticker: str = '005930', src: str = 'github', period: int = 5, meta = None):
        super().__init__(ticker=ticker, src=src, period=period, meta=meta)


class metadata:

    def __init__(self):
        self.master = 'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/'
        self._krx_index_raw = pd.DataFrame()
        self._market_data = pd.DataFrame()
        return

    @property
    def krx_index_raw(self) -> pd.DataFrame:
        if self._krx_index_raw.empty:
            self._krx_index_raw = pd.read_csv(f'{self.master}warehouse/meta-index.csv', encoding='utf-8')
        return self._krx_index_raw

    @property
    def krx_index(self) -> pd.DataFrame:
        objs = []
        index_raw = self.krx_index_raw.set_index(keys='종목코드')
        for t, name in [('KS', '코스피'), ('KQ', '코스닥'), ('KX', 'KRX'), ('TM', '테마')]:
            obj = index_raw[index_raw['거래소'] == t].copy()
            obj.index.name = f'{name}코드'
            objs.append(obj.rename(columns={'종목명': f'{name}종류'}).drop(columns=['거래소']).reset_index(level=0))
        frm = pd.concat(objs=objs, axis=1).fillna('-')
        return frm

    @property
    def market_data(self) -> pd.DataFrame:
        if self._market_data.empty:
            base = pd.read_csv(f'{self.master}warehouse/market/market.csv', encoding='utf-8', index_col='종목코드')
            wics = pd.read_csv(f'{self.master}warehouse/group/WICS.csv', encoding='utf-8', index_col='종목코드')
            wi26 = pd.read_csv(f'{self.master}warehouse/group/WI26.csv', encoding='utf-8', index_col='종목코드')

            wics = wics.drop(columns=['종목명']).rename(columns={'산업':'WICS(대)', '섹터':'WICS(소)'})
            wi26 = wi26.drop(columns=['종목명']).rename(columns={'섹터':'WI26'})
            self._market_data = pd.concat([base, wics, wi26], axis=1)
            self._market_data.index = self._market_data.index.astype(str).str.zfill(6)
        return self._market_data

if __name__ == "__main__":
    api = metadata()
    # print(api.krx_index)
    print(api.market_data)
