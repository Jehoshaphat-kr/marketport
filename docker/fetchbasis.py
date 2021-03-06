from datetime import datetime
from pykrx import stock
import pandas as pd
import urllib.request as req
import json, os


__root__ = os.path.dirname(os.path.dirname(__file__))
class dock:
    
    url_stk = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    url_etf = 'https://finance.naver.com/api/sise/etfItemList.nhn'
    warehouse = os.path.join(__root__, 'warehouse')

    today = datetime.today().date()
    stocks = pd.DataFrame()
    etfs = pd.DataFrame()

    def __init__(self, date:datetime=None):
        print("=" * 50)
        print("|" + " " * 14 + "메타데이터 업데이트" + " " * 15 + "|")
        print("=" * 50)

        self.today = date if date else self.today
        return

    def __trig__(self) -> None:
        """
        실행 트리거: 상장일에 따라 업데이트 여부 결정
        :return:
        """
        self.stocks = pd.read_html(self.url_stk, header=0)[0]
        self.stocks = self.stocks[['회사명', '종목코드', '상장일']].rename(columns={'회사명': '종목명'})
        return

    def update(self) -> None:
        """
        메타 데이터 업데이트
        :return:
        """
        self.__trig__()
        self.rename_stock()
        self.update_stock()
        self.update_etf()
        self.update_index()

        data = pd.concat(objs=[self.stocks, self.etfs], axis=0)
        data.to_csv(path_or_buf=os.path.join(self.warehouse, 'meta-stock.csv'), encoding='utf-8', index=False)
        self.check()
        return

    def rename_stock(self) -> None:
        """
        KRX 고유 명명법 --> 범용 종목명 변경
        :return:
        """
        print("Proc 01: 종목 이름 변경 대상 확인 중...")
        self.stocks.set_index(keys='종목코드', inplace=True)
        self.stocks.index = self.stocks.index.astype(str).str.zfill(6)

        rename = pd.read_csv(
            filepath_or_buffer=os.path.join(self.warehouse, 'group/handler/RENAME.csv'),
            encoding='utf-8',
            index_col='종목코드'
        )
        rename.index = rename.index.astype(str).str.zfill(6)
        dividend = rename[rename['특징'] == '우선주'].copy()
        rename.drop(index=dividend.index, inplace=True)

        renamed = self.stocks[self.stocks.index.isin(rename.index)].copy()
        renamed.drop(columns=['종목명'], inplace=True)
        renamed = renamed.join(rename[['종목명']], how='left')

        self.stocks.drop(index=renamed.index, inplace=True)
        self.stocks = pd.concat(objs=[self.stocks, renamed, dividend[['종목명']]], axis=0)
        return

    def update_stock(self) -> None:
        """
        KRX 거래소 기준 주식 전 종목 수집
        :return:
        """
        today = self.today.strftime("%Y%m%d")

        print("Proc 02: 종목별 거래소 설정 확인 중...")
        ks = stock.get_market_ticker_list(today, market='KOSPI')
        kq = stock.get_market_ticker_list(today, market='KOSDAQ')
        trader = pd.concat(objs=[
            pd.Series(data=['KS'] * len(ks), index=ks, name='거래소'),
            pd.Series(data=['KQ'] * len(kq), index=kq, name='거래소')
        ], axis=0)
        self.stocks = self.stocks.join(trader, how='left')

        print("Proc 03: 종가 및 시가총액 다운로드 중...")
        cap = stock.get_market_cap_by_ticker(self.today.strftime("%Y%m%d"))[['종가', '시가총액']]
        self.stocks = self.stocks.join(cap, how='left')
        self.stocks = self.stocks[~self.stocks['거래소'].isna()].copy()
        self.stocks.index.name = '종목코드'
        self.stocks.reset_index(level=0, inplace=True)
        return

    def update_etf(self) -> None:
        """
        네이버 기준 ETF 전 종목 수집
        :return:
        """
        print("Proc 04: ETF 리스트 확인 중...")
        fetch = pd.DataFrame(json.loads(req.urlopen(self.url_etf).read().decode('cp949'))['result']['etfItemList'])
        fetch['거래소'] = 'ETF'
        self.etfs = fetch[['itemcode', 'itemname', 'nowVal', 'marketSum', '거래소']].copy()
        self.etfs.rename(
            columns=dict(zip(
                ['itemcode', 'itemname', 'nowVal', 'marketSum'], 
                ['종목코드', '종목명', '종가', '시가총액']
            )), inplace=True
        )
        self.etfs['시가총액'] = self.etfs['시가총액'] * 100000000
        return

    def update_index(self) -> None:
        """
        KRX 지수 종류 데이터프레임
        :return:
        """
        print("Proc 05: KRX 지수 종류 수집 중...")
        ks_tickers = stock.get_index_ticker_list(market='KOSPI')
        ks_indices = [stock.get_index_ticker_name(ticker) for ticker in ks_tickers]
        ks_kind = ['KS'] * len(ks_tickers)
        ks = pd.DataFrame(data={'종목코드':ks_tickers, '종목명':ks_indices, '거래소':ks_kind})

        kq_tickers = stock.get_index_ticker_list(market='KOSDAQ')
        kq_indices = [stock.get_index_ticker_name(ticker) for ticker in kq_tickers]
        kq_kind = ['KQ'] * len(kq_tickers)
        kq = pd.DataFrame(data={'종목코드': kq_tickers, '종목명': kq_indices, '거래소': kq_kind})

        kx_tickers = stock.get_index_ticker_list(market='KRX')
        kx_indices = [stock.get_index_ticker_name(ticker) for ticker in kx_tickers]
        kx_kind = ['KX'] * len(kx_tickers)
        kx = pd.DataFrame(data={'종목코드': kx_tickers, '종목명': kx_indices, '거래소': kx_kind})

        tm_tickers = stock.get_index_ticker_list(market='테마')
        tm_indices = [stock.get_index_ticker_name(ticker) for ticker in tm_tickers]
        tm_kind = ['TM'] * len(tm_tickers)
        tm = pd.DataFrame(data={'종목코드': tm_tickers, '종목명': tm_indices, '거래소': tm_kind})

        frm = pd.concat(objs=[ks, kq, kx, tm], axis=0)
        frm.to_csv(os.path.join(self.warehouse, 'meta-index.csv'), index=False, encoding='utf-8')
        return

    def check(self) -> None:
        """
        메타데이터 이상 여부 확인
        :return:
        """
        print("Proc 06: 메타데이터 이상 여부 확인...")
        data = pd.read_csv(os.path.join(self.warehouse, 'meta-stock.csv'), encoding='utf-8', index_col='종목코드')
        data.index = data.index.astype(str).str.zfill(6)
        duplicate = data.index.value_counts()
        duplicate = duplicate[duplicate >= 2].index
        print("  - 중복 발생 여부 확인 중...", end=" ")
        if not duplicate.empty:
            print('')
            print(data[data.index.isin(duplicate)])
            print('-' * 70)
        else:
            print('없음')
        print("  - 종가/시가총액 누락 여부 확인 중...", end=" ")
        na = data[data['종가'].isna() | data['시가총액'].isna()]
        if na.empty:
            print("없음")
        else:
            print('')
            print(na)
            print('-' * 70)
        return


if __name__ == "__main__":
    ''' 메타 데이터 관리 '''
    docker = dock(
        # date=datetime(2021, 10, 13)
    )
    docker.update()