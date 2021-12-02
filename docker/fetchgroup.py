from datetime import datetime
import pandas as pd
import os, time, requests


__root__ = os.path.dirname(os.path.dirname(__file__))
wics = {
    'G1010': '에너지',
    'G1510': '소재',
    'G2010': '자본재',
    'G2020': '상업서비스와공급품',
    'G2030': '운송',
    'G2510': '자동차와부품',
    'G2520': '내구소비재와의류',
    'G2530': '호텔,레스토랑,레저 등',
    'G2550': '소매(유통)',
    'G2560': '교육서비스',
    'G3010': '식품과기본식료품소매',
    'G3020': '식품,음료,담배',
    'G3030': '가정용품과개인용품',
    'G3510': '건강관리장비와서비스',
    'G3520': '제약과생물공학',
    'G4010': '은행',
    'G4020': '증권',
    'G4030': '다각화된금융',
    'G4040': '보험',
    'G4050': '부동산',
    'G4510': '소프트웨어와서비스',
    'G4520': '기술하드웨어와장비',
    'G4530': '반도체와반도체장비',
    'G4535': '전자와 전기제품',
    'G4540': '디스플레이',
    'G5010': '전기통신서비스',
    'G5020': '미디어와엔터테인먼트',
    'G5510': '유틸리티'
}

wi26 = {
    'WI100': '에너지',
    'WI110': '화학',
    'WI200': '비철금속',
    'WI210': '철강',
    'WI220': '건설',
    'WI230': '기계',
    'WI240': '조선',
    'WI250': '상사,자본재',
    'WI260': '운송',
    'WI300': '자동차',
    'WI310': '화장품,의류',
    'WI320': '호텔,레저',
    'WI330': '미디어,교육',
    'WI340': '소매(유통)',
    'WI400': '필수소비재',
    'WI410': '건강관리',
    'WI500': '은행',
    'WI510': '증권',
    'WI520': '보험',
    'WI600': '소프트웨어',
    'WI610': 'IT하드웨어',
    'WI620': '반도체',
    'WI630': 'IT가전',
    'WI640': '디스플레이',
    'WI700': '통신서비스',
    'WI800': '유틸리티',
}

class dock:

    def __init__(self):
        self.dir_storage = os.path.join(__root__, 'warehouse/group')
        self.dir_handler = os.path.join(__root__, 'warehouse/group/handler')

        self.meta = pd.read_csv(
            filepath_or_buffer=os.path.join(os.path.join(__root__, 'warehouse'), 'meta-stock.csv'),
            encoding='utf-8',
            index_col='종목코드'
        )
        self.__date__ = ''
        return

    @property
    def date_src(self) -> str:
        """
        WICS/WI26 소스 최근 데이터 날짜
        :return:
        """
        if not self.__date__:
            source = requests.get(url='http://www.wiseindex.com/Index/Index#/G1010.0.Components').text
            i_ = source.find("기준일")
            _i = source[i_:].find("</p>")
            self.__date__ = datetime.strptime(source[i_ + 6: i_ + _i], "%Y.%m.%d").strftime("%Y%m%d")
        return self.__date__

    def fetch(self, code:str) -> pd.DataFrame:
        """
        개별 지수(산업, 섹터) 다운로드
        :param code:
        :return:
        """
        response = requests.get(
            url=f'http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={self.date_src}&sec_cd={code}'
        )
        if response.status_code == 200:
            try:
                json = response.json()
                data = [
                    [_['CMP_CD'], _['CMP_KOR'], _['SEC_NM_KOR'], _['IDX_NM_KOR'][5:]] for _ in json['list']
                ]
                return pd.DataFrame(data=data, columns=['종목코드', '종목명', '산업', '섹터'])
            except ConnectionError as e:
                print(f'\t- Parse error while fetching {code}')
        else:
            print(f'\t- Connection error while fetching {code}')
        return pd.DataFrame()

    def update_group(self, kind:str):
        """
        지수 그룹 업데이트
        :param kind: wics/wi26
        :return:
        """
        print("=" * 50)
        print("|" + " " * 10 + f"{kind.upper()} 산업/업종 분류 다운로드" + " " * 10 + "|")
        print("=" * 50)
        print(f'기준 날짜: {self.date_src} <-- Source')
        if not kind in ['wics', 'wi26']:
            raise ValueError(f'Argument kind must passed either "wics" or "wi26", but {kind} is passed.')
        meta = wics if kind == 'wics' else wi26

        group = pd.DataFrame()
        done = list(meta.keys())
        while done:
            code = done[0]
            print(f'{100 * (list(meta.keys()).index(code) + 1) / len(meta):.2f}% :: {code} {meta[code]}')
            df = self.fetch(code=code)

            if not df.empty:
                group = group.append(df, ignore_index=True)
                done.remove(code)
                continue
            time.sleep(0.5)

        if kind == 'wi26':
            group.drop(columns=['산업'], inplace=True)
            # group.rename(columns={'산업':'섹터'}, inplace=True)
        group.to_csv(os.path.join(self.dir_storage, f'{kind.upper()}.csv'), index=False)
        return

    def update_theme(self) -> None:
        """
        테마 파일 CSV 변환
        :return:
        """
        print("** 테마 분류 업데이트 **")
        meta = self.meta.reset_index(level=0).copy()
        meta.set_index(keys='종목명', inplace=True)
        meta['종목코드'] = meta['종목코드'].astype(str).str.zfill(6)

        theme = pd.read_excel(os.path.join(self.dir_handler, 'THEME.xlsx'), index_col='종목명')
        theme.drop(columns=['출처'], inplace=True)
        theme = theme.join(meta[['종목코드']], how='left')

        no_ticker = theme[theme['종목코드'].isna()]
        if not no_ticker.empty:
            print("종목코드 미상 항목 발생: 업데이트 중단")
            print(no_ticker)
            print("RENAME.csv @handler 업데이트 필요")
        else:
            theme.reset_index(level=0, inplace=True)
            theme.to_csv(os.path.join(self.dir_storage, 'THEME.csv'), index=False, encoding='utf-8')
        print("-" * 70)
        return

    def check_etf(self) -> None:
        """
        ETF 최신화 여부 확인
        :return: 
        """
        print("=" * 50)
        print("|" + " " * 15 + f"ETF 분류 다운로드" + " " * 16 + "|")
        print("=" * 50)
        etf_online = self.meta[self.meta['거래소'] == 'ETF'].copy()
        etf_offline = pd.read_excel(os.path.join(self.dir_handler, 'TDATETF.xlsx'), index_col='종목코드')

        to_be_delete = etf_offline[~etf_offline.index.isin(etf_online.index)]
        to_be_update = etf_online[~etf_online.index.isin(etf_offline.index)]

        ''' NOTICE '''
        for kind, frm in [('삭제', to_be_delete), ('추가', to_be_update)]:
            print(f"▷ ETF 분류 {kind} 필요 항목: {'없음' if frm.empty else '있음'}")
            if not frm.empty:
                print(frm)
                if kind == '추가':
                    os.startfile(self.dir_handler)
            print("-" * 50)
        return

    def update_etf(self) -> None:
        """
        ETF 수기 파일 CSV 변환
        :return:
        """
        etf_offline = pd.read_excel(os.path.join(self.dir_handler, 'TDATETF.xlsx'))
        etf_offline.to_csv(os.path.join(self.dir_storage, 'ETF.csv'), index=False, encoding='utf-8')
        return


if __name__ == "__main__":

    docker = dock()

    ''' WICS/WI26 '''
    docker.update_group(kind='wi26')
    docker.update_group(kind='wics')

    ''' ETF '''
    docker.check_etf()
    docker.update_etf()

    ''' THEME '''
    # docker.update_theme()
