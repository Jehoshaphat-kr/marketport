from pykrx import stock
from datetime import datetime
import pandas as pd
import os, time


etf_list = [
    '305540',  # TIGER 2차전지테마
    '091230',  # TIGER 반도체
    '139280',  # TIGER 경기방어
    '228800',  # TIGER 여행레저
    '228790',  # TIGER 화장품
    '143860',  # TIGER 헬스케어
    '091220',  # TIGER 은행
]
#
#     '305720',  # KODEX 2차전지산업
#     '091160',  # KODEX 반도체
#     '091180',  # KODEX 자동차
#     '325010',  # KODEX Fn성장
#     '244620',  # KODEX 모멘텀Plus
#     '117700',  # KODEX 건설
#     '117680',  # KODEX 철강
#     '117460',  # KODEX 에너지화학
#     '140710',  # KODEX 운송
#     '266390',  # KODEX 경기소비재
#     '244580',  # KODEX 바이오
#     '091170',  # KODEX 은행
#     '102970',  # KODEX 증권
#     '140700',  # KODEX 보험
#
#     '140570',  # KBSTAR 수출주
#     '367770',  # KBSTAR Fn수소경제테마
# ]

__root__ = os.path.dirname(os.path.dirname(__file__))
class dock:

    warehouse = os.path.join(__root__, 'warehouse')
    today = datetime.today()
    meta = pd.read_csv(
        filepath_or_buffer=os.path.join(warehouse, 'meta-stock.csv'),
        encoding='utf-8',
        index_col='종목코드'
    )
    meta.index = meta.index.astype(str).str.zfill(6)
    proc = 1

    def __init__(self, date:datetime=None):
        print("=" * 50)
        print("|" + " " * 11 + "ETF 보유 종목 현황 업데이트" + " " * 10 + "|")
        print("=" * 50)

        self.today = date if date else self.today
        print(f'PROP 날짜: {self.today.strftime("%Y-%m-%d")}')
        return

    def initialize(self, ticker):
        """
        최초 ETF 구성 종목 다운로드 (최근 3개월 치)
        :param ticker:
        :return:
        """
        print(f"Proc {str(self.proc).zfill(2)}: [{ticker}] {self.get_name(ticker)} 초기 구성 비율 다운로드 중...")
        self.proc += 1

        span = pd.read_csv(
            filepath_or_buffer=os.path.join(self.warehouse, 'price/005930.csv'),
            index_col='날짜',
            encoding='utf-8'
        ).tail(252).index
        span = [datetime.strptime(date, "%Y-%m-%d") for date in span]

        objs = []
        for n, date in enumerate(span):
            fetch = stock.get_etf_portfolio_deposit_file(ticker=ticker, date=date.strftime("%Y%m%d"))
            if '' in fetch.index:
                fetch.drop(index=[''], inplace=True)
            fetch.index = [self.get_name(ticker=t) if t in docker.meta.index else t for t in fetch.index]
            objs.append(
                pd.DataFrame(
                    data=dict(zip(fetch.index, fetch['비중'])),
                    index=[date]
                )
            )
            if not (n+1) % 10:
                time.sleep(3)

        frm = pd.concat(objs=objs, axis=0)
        frm.index.name = '날짜'
        frm.reset_index(level=0, inplace=True)
        frm.to_csv(os.path.join(self.warehouse, f'deposit/{ticker}.csv'), index=False, encoding='utf-8')
        return

    def get_name(self, ticker) -> str:
        """
        ETF 이름 반환
        :param ticker:
        :return:
        """
        return self.meta.loc[ticker, '종목명']

    def update(self) -> None:
        """
        ETF 보유 종목 현황 다운로드
        :return:
        """
        for ticker in etf_list:
            if not os.path.isfile(os.path.join(self.warehouse, f'deposit/{ticker}.csv')):
                self.initialize(ticker=ticker)

            prev = pd.read_csv(
                filepath_or_buffer=os.path.join(self.warehouse, f'deposit/{ticker}.csv'),
                encoding='utf-8',
                index_col='날짜'
            )
            if prev.index[-1] == self.today.strftime("%Y-%m-%d"):
                continue

            print(f"Proc {str(self.proc).zfill(2)}: [{ticker}] {self.get_name(ticker)} 업데이트 중...")
            self.proc += 1

            fetch = stock.get_etf_portfolio_deposit_file(ticker=ticker, date=self.today.strftime("%Y%m%d"))
            if '' in fetch.index:
                fetch.drop(index=[''], inplace=True)
            fetch.index = [self.get_name(ticker=t) if t in self.meta.index else t for t in fetch.index]

            curr = pd.concat(
                objs=[
                    prev,
                    pd.DataFrame(
                        data=dict(zip(fetch.index, fetch['비중'])),
                        index=[self.today.strftime("%Y-%m-%d")]
                    )
                ], axis=0
            )
            for col in curr.columns:
                curr[col] = round(curr[col], 2)
            curr.index.name = '날짜'
            curr.reset_index(level=0, inplace=True)
            curr.to_csv(os.path.join(self.warehouse, f'deposit/{ticker}.csv'), encoding='utf-8', index=False)
        return

    def update_meta(self) -> None:
        """
        ETF 기본 정보 입력 업데이트
        :return:
        """
        print(f"Proc {str(self.proc).zfill(2)}: 메타 데이터 생성 중...")
        self.proc += 1

        xlsx = pd.read_excel(
            io=os.path.join(self.warehouse, 'deposit/handler/guide.xlsx')
        )
        xlsx['종목코드'] = xlsx['종목코드'].astype(str).str.zfill(6)
        xlsx.to_csv(
            path_or_buf=os.path.join(self.warehouse, 'deposit/meta.csv'),
            encoding='utf-8',
            index=False
        )
        print(xlsx)
        return



if __name__ == "__main__":

    docker = dock(
        # date=datetime(2021, 10, 8)
    )
    docker.update()
    # docker.update_meta()
