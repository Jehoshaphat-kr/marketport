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
        print("PROP. 날짜 : {}".format(self.today.strftime("%Y-%m-%d")))
        return

    def __trig__(self) -> bool:
        """
        실행 트리거: 상장일에 따라 업데이트 여부 결정
        :return:
        """
        today = self.today.strftime("%Y-%m-%d")
        fetch = pd.read_html(self.url_stk, header=0)[0]
        fetch = fetch[['회사명', '종목코드', '상장일']].rename(columns={'회사명': '종목명'})
        if fetch[fetch['상장일'] == today].empty:
            return False
        self.stocks = fetch.copy()
        return True

    def update(self) -> None:
        """
        메타 데이터 업데이트
        :return:
        """
        if self.__trig__():
            self.stock_rename()
            self.stock_update()
            self.etf_update()

            data = pd.concat(objs=[self.stocks, self.etfs], axis=0)
            data.to_csv(os.path.join(self.warehouse, 'meta-stock.csv'), encoding='utf-8', index=False)

            self.check_update()
            print("완료")
        else:
            print("업데이트 대상 없음")
        return

    def stock_rename(self) -> None:
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

    def stock_update(self) -> None:
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

    def etf_update(self) -> None:
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

    def check_update(self) -> None:
        """
        메타데이터 이상 여부 확인
        :return:
        """
        print("Proc 05: 메타데이터 이상 여부 확인...")
        data = pd.read_csv(os.path.join(self.warehouse, 'meta-stock.csv'), encoding='utf-8', index_col='종목코드')
        data.index = data.index.astype(str).str.zfill(6)
        duplicate = data.index.value_counts()
        duplicate = duplicate[duplicate >= 2].index
        if not duplicate.empty:
            print("중목 항목 발생")
            print(data[data.index.isin(duplicate)])
            print('-' * 70)
        return


if __name__ == "__main__":
    ''' 메타 데이터 관리 '''
    docker = dock(
        date=datetime(2021, 10, 8)
    )
    docker.update()
