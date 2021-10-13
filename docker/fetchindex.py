from datetime import datetime
from pykrx import stock
import pandas as pd
import os, time

__root__ = os.path.dirname(os.path.dirname(__file__))
class dock:
    dir_warehouse = os.path.join(__root__, 'warehouse')
    dir_storage = os.path.join(__root__, 'warehouse/index')
    tic = datetime(2010, 1, 2).strftime("%Y%m%d")

    def __init__(self, date:datetime=None):
        print("=" * 50)
        print("|" + " " * 15 + "KRX 지수 업데이트" + " " * 16 + "|")
        print("=" * 50)
        today = datetime.today() if not date else date
        self.toc = today.strftime("%Y%m%d")
        print(f"PROP 날짜: {today.strftime('%Y-%m-%d')}")

        self.frm = self.__frm__()
        self.frm.to_csv(os.path.join(self.dir_warehouse, 'meta-index.csv'), index=False, encoding='utf-8')
        return

    def __frm__(self) -> pd.Series:
        """
        KRX 지수 종류 데이터프레임
        :return:
        """
        print("Proc 01: KRX 지수 종류 수집 중...")
        ks_tickers = stock.get_index_ticker_list(market='KOSPI')
        ks_indices = [stock.get_index_ticker_name(ticker) for ticker in ks_tickers]
        ks_kind = ['KS'] * len(ks_tickers)
        ks = pd.DataFrame(data={'지수코드':ks_tickers, '지수명':ks_indices, '종류':ks_kind})

        kq_tickers = stock.get_index_ticker_list(market='KOSDAQ')
        kq_indices = [stock.get_index_ticker_name(ticker) for ticker in kq_tickers]
        kq_kind = ['KQ'] * len(kq_tickers)
        kq = pd.DataFrame(data={'지수코드': kq_tickers, '지수명': kq_indices, '종류': kq_kind})

        kx_tickers = stock.get_index_ticker_list(market='KRX')
        kx_indices = [stock.get_index_ticker_name(ticker) for ticker in kx_tickers]
        kx_kind = ['KX'] * len(kx_tickers)
        kx = pd.DataFrame(data={'지수코드': kx_tickers, '지수명': kx_indices, '종류': kx_kind})
        return pd.concat(objs=[ks, kq, kx], axis=0)

    def fetch(self, index: str, tic:str='') -> pd.DataFrame:
        """
        단일 지수코드에 대한 가격 데이터프레임 다운로드
        :param index: 지수코드
        :param tic: 시작 시간
        :return:
        """
        tic = self.tic if not tic else tic
        error_message = ''
        for trial_cnt in range(5):
            try:
                data = stock.get_index_ohlcv_by_date(fromdate=tic, todate=self.toc, ticker=index)
                data.reset_index(level=0, inplace=True)
                return data
            except Exception as e:
                time.sleep(5)
                if trial_cnt == 4:
                    error_message = e
        print("[{}] : ".format(index), error_message)
        return pd.DataFrame()
    
    def update(self, debug:bool=False):
        """
        전체 지수코드에 대한 지수 데이터프레임 다운로드 후 저장
        :param debug:
        :return:
        """
        print("Proc 02: 전체 지수 업데이트 중..")
        for n, index in enumerate(self.frm['지수코드']):
            if debug: print('    {:3.2f}%: {}'.format(100 * (n + 1) / len(self.frm), index))
            data = self.fetch(index=index)
            data.to_csv(os.path.join(self.dir_storage, '{}.csv'.format(index)), encoding='utf-8', index=False)
            if not (n+1) % 12:
                time.sleep(3)

if __name__ == "__main__":
    docker = dock()
    docker.update(debug=True)