from datetime import datetime
from pykrx import stock
import pandas as pd
import os, codecs, jsmin


__root__ = os.path.dirname(os.path.dirname(__file__))
class frame:

    time_span = ['R1D', 'R1W', 'R1M', 'R3M', 'R6M', 'R1Y']
    multiples = ['PER', 'PBR', 'DIV']
    # url_g = 'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/group/{}.csv'
    # url_m = 'https://raw.githubusercontent.com/Jehoshaphat-kr/marketport/master/warehouse/market/{}.csv'
    kind = ''
    code = ''
    name = ''

    def __init__(self, date: datetime = None):
        """
        MARKET MAP 데이터프레임 생성
        :param date: 날짜 in format %Y%m%d
        """
        self.date = datetime.today().date() if not date else date.date()
        self.__map__ = pd.DataFrame()
        self.__bar__ = pd.DataFrame()
        return

    def __attr__(self, kind:str, code:str='') -> None:
        """
        시장 지도 속성 설정
        - 지도 유형 별 생성(map_type) : WICS(산업분류), WI26(업종분류), ETF(ETF), THEME(테마)
        :param kind: 지도 유형
        :param code: WICS/WI26 한정 유효
            - PYKRX 기준 지수(Index) 코드
            - Default: 0 전체 시장
              1002 코스피 대형주
              1003 코스피 중형주
              1004 코스피 소형주
              1028 코스피 200
              2002 코스닥 대형주
              2003 코스닥 중형주
              2203 코스닥 150
        :return:
        """
        self.kind = kind
        self.code = code
        if kind in ["WICS", "WI26"] and not code:
            self.name = "전체"
        elif kind in ["WICS", "WI26"] and code:
            self.name = stock.get_index_ticker_name(str(code))
        elif kind == "ETF":
            self.name = kind
        elif kind == 'THEME':
            self.name = "테마"
        return

    @property
    def mapframe(self) -> pd.DataFrame:
        """
        시장 지도 데이터프레임
        :return:
        """
        _01_raw_data = self.__base__()
        _02_pre_data = self.__pre__(data=_01_raw_data)
        _03_run_data = self.__run__(data=_02_pre_data)
        self.__map__ = self.__post__(data=_03_run_data)
        return self.__map__

    @property
    def barframe(self) -> pd.DataFrame:
        """
        막대 그래프 데이터프레임
        :return: 
        """
        return self.__map__[self.__map__["종목코드"].isin(self.__bar__)].copy()

    def cindex(self, data) -> dict:
        """
        색상 기준 수익률 리스트 산출
        :return: 수익률 기간별 기준 수익률 (색상 산출용)
        """
        risk_free = 2.0  # 연간 무위험 수익
        steadiness = {'R1Y': risk_free, 'R6M': risk_free * 0.5, 'R3M': risk_free * 0.25,
                      'R1M': risk_free * 0.08, 'R1W': risk_free * 0.02, 'R1D': risk_free * 0.005}

        ''' 기간별 구간 구분 및 색상 설정 '''
        range = {}
        for gap in steadiness.keys():
            if not gap in data.columns:
                continue
            steady_point = steadiness[gap]
            sr = data[gap].sort_values(ascending=False)
            lower = sr[sr < -steady_point]
            upper = sr[sr > steady_point]
            range[gap] = [
                lower.values[int(0.66 * len(lower))],
                lower.values[int(0.33 * len(lower))],
                -steady_point, steady_point,
                upper.values[int(0.66 * len(upper))],
                upper.values[int(0.33 * len(upper))]
            ]
        return range

    def __base__(self) -> pd.DataFrame:
        """
        시장 지도 기저 데이터프레임 생성
        :return:
        """
        return pd.read_csv(
            filepath_or_buffer=os.path.join(__root__, f'warehouse/group/{self.kind}.csv'),
            encoding='utf-8', index_col='종목코드'
        ).join(
            other=pd.read_csv(
                filepath_or_buffer=os.path.join(__root__, 'warehouse/market/market.csv'),
                encoding='utf-8', index_col="종목코드"
            ).drop(columns=['종목명']), how='left'
        )

    def __pre__(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        시장 지도 데이터 전처리
        :param data:
        :return:
        """
        frm = data.copy()
        ''' 종가 정보 수집 불가 항목(상장 폐지) 삭제 '''
        frm.drop(index=frm.loc[data['종가'].isna()].index, inplace=True)

        ''' 변동성 정보 삭제 '''
        frm.drop(columns=[col for col in frm.columns if col.startswith('V') and col[1].isdigit()], inplace=True)

        ''' 종목코드 형식 변경 '''
        frm.index = frm.index.astype(str).str.zfill(6)

        ''' 종가 형식 변경: 실수형에서 통화 단위 '''
        frm['종가'] = frm['종가'].apply(lambda p: '{:,}'.format(int(p)))

        ''' 시장 지도 크기 정보 추가 '''
        frm["크기"] = frm["시가총액"] / 100000000

        ''' 산업/업종 지도 한정, 구속 조건 반영(시가총액/거래소) '''
        if self.kind in ["WICS", "WI26"] and self.code:
            tickers = stock.get_index_portfolio_deposit_file(str(self.code))
            frm = frm[frm.index.isin(tickers)].copy()

        ''' 개별주 한정, 코스닥 인식자(*) 삽입 '''
        if not self.kind == "ETF" and not self.code:
            frm['종목명'] = [
                name + '*' if trade == 'KQ' else name
                for name, trade in zip(frm['종목명'], frm['거래소'])
            ]

        ''' 시가총액 기준 조정(미달 종목 제거) '''
        lim = 3000 if self.kind in ["WICS", "WI26"] and not self.code else 1
        frm = frm[frm['크기'] > lim].copy()
        return frm

    def __run__(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        시장 지도 프레임 분류법 1차원 축소
        :return:
        """
        data.reset_index(level=0, inplace=True)
        levels = ["종목코드", "섹터", "산업"]
        if not self.kind in ["WICS", "ETF"]:
            levels.remove("산업")

        frm = pd.DataFrame()
        for index, level in enumerate(levels):
            if not index:
                ''' 일반 종목 전체 삽입 '''
                branch = data.copy()
                branch.rename(columns={'섹터': '분류'}, inplace=True)
                branch.drop(
                    columns=['거래소', '산업'] if '산업' in branch.columns else ['거래소'], inplace=True
                )
            else:
                ''' 상위 분류(일반 종목 분류) 데이터 삽입 '''
                branch = pd.DataFrame()
                layer = data.groupby(levels[index:]).sum().reset_index()

                ''' 
                기본형 데이터 삽입 
                종목코드    종목명    분류    크기 
                '''
                identifier = self.kind if not self.code else self.kind + self.name
                branch["종목코드"] = layer[level] + "_" + identifier
                branch["종목명"] = layer[level]
                branch["분류"] = layer[levels[index + 1]] if index < len(levels) - 1 else self.name
                branch["크기"] = layer[["크기"]]

                '''
                분류별 수익률, 배수 산출 (시가총액 가중)
                - 해당 분류 내 종목의 펙터를 시가총액으로 가중
                - 가중된 펙터 합을 분류 별 펙터로 산정
                                              (cap x yield)
                - f(group, yield, cap) = ∑ -------------------
                                             ∑(cap of group)

                '''
                for group_name in branch['종목명']:
                    grouped = data[data[level] == group_name]
                    for factor in self.time_span + self.multiples:
                        if factor == "DIV":
                            branch.loc[branch['종목명'] == group_name, factor] = (
                                    grouped[factor].sum() / len(grouped)
                            ) if not grouped.empty else 0
                        else:
                            if factor == 'PER':
                                grouped = grouped[grouped['PER'] != 0].copy()
                            branch.loc[branch["종목명"] == group_name, factor] = (
                                    grouped[factor] * grouped['크기'] / grouped['크기'].sum()
                            ).sum()

                if level == "섹터":
                    self.__bar__ = branch.copy()["종목코드"]
            frm = frm.append(branch, ignore_index=True)

            if index == len(levels) - 1:
                _cover = {
                    '종목코드': self.name + '_' + self.kind.lower(),
                    '종목명': self.name,
                    '분류': '',
                    '크기': data['크기'].sum()
                }
                frm = frm.append(pd.DataFrame(pd.DataFrame(_cover, index=[0])), ignore_index=True)

        ''' 시가총액 형식 변경: 정수형에서 통화 단위 '''
        cap = frm["크기"].astype(int).astype(str)
        frm['시가총액'] = cap.apply(lambda v: v + "억" if len(v) < 5 else v[:-4] + '조 ' + v[-4:] + '억')

        ''' 수익률 소수 둘째 자리 반올림 적용 '''
        for key in self.time_span + self.multiples:
            frm[key] = frm[key].apply(lambda v: round(v, 2))

        ''' 중복 종목명 제거 '''
        if not self.kind == "THEME":
            frm.drop_duplicates(subset=['종목명'], keep='last', inplace=True)
        return frm

    def __post__(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        색상 정의 및 데이터 후처리
        :return:
        """
        scale = ['#F63538', '#BF4045', '#8B444E', '#414554', '#35764E', '#2F9E4F', '#30CC5A']  # Low <---> High
        index = self.cindex(data=data)
        ''' 수익률 기간별 색상 선정 '''
        __frm__ = data.copy()
        __frm__.set_index(keys=['종목코드', '분류'] if self.kind == "THEME" else ['종목코드'], inplace=True)

        df_color = pd.DataFrame(index=__frm__.index)
        for period in index.keys():
            _returns = __frm__[period].copy()
            _returns.fillna(0, inplace=True)

            sr = pd.cut(_returns, bins=[_returns.min()] + index[period] + [_returns.max()], labels=scale, right=True)
            sr.name = "C" + period
            df_color = df_color.join(sr.astype(str), how='left')
        __frm__ = __frm__.join(df_color, how='left')

        if self.kind == 'ETF':
            for col in [key for key in __frm__.columns if 'C' in key]:
                __frm__.at[__frm__.index[-1], col] = '#C8C8C8'
            __frm__.reset_index(level=['종목코드', '분류'] if self.kind == "THEME" else ['종목코드'], inplace=True)
            __frm__.drop(columns=['PER', 'PBR', 'DIV'], inplace=True)
            return __frm__

        ''' 배수(Multiple) 색상 선정 '''
        for multiple in self.multiples:
            color_label = scale if multiple == "DIV" else scale[::-1]
            trunk = __frm__[__frm__[multiple] != 0].sort_values(by=multiple, ascending=False).copy()
            value = trunk[multiple].dropna().copy()

            thres = [value[int(len(value) / 7) * i] for i in range(len(scale))] + [value[-1]]
            color = pd.cut(value, bins=thres[::-1], labels=color_label, right=True)
            color.name = "C" + multiple
            __frm__ = __frm__.join(color.astype(str), how='left')
            __frm__[color.name].fillna(scale[0] if multiple == "DIV" else scale[3], inplace=True)

        for col in [key for key in __frm__.columns if 'C' in key]:
            __frm__.at[__frm__.index[-1], col] = '#C8C8C8'
        __frm__.reset_index(level=['종목코드', '분류'] if self.kind == "THEME" else ['종목코드'], inplace=True)
        __frm__['PER'] = __frm__['PER'].apply(lambda val: val if not val == 0 else 'N/A')
        return __frm__


class map2js(frame):
    labels = {}
    covers = {}
    ids = {}
    bar = {}

    suffix = codecs.open(filename=os.path.join(__root__, 'docker/handler/map-suffix.js'), mode='r', encoding='utf-8').read()
    datum = pd.DataFrame(columns=['종목코드'])
    cover = list()

    def collect(self):
        print("=" * 50)
        print("|" + " " * 10 + "시장 지도 데이터 프레임 생성" + " " * 10 + "|")
        print("=" * 50)
        print(f"PROP 날짜: {self.date.strftime('%Y-%m-%d')}")

        maps = [
            ["WICS", '', "indful"],
            ["WICS", '1002', "indksl"], ["WICS", '1003', "indksm"], ["WICS", '1004', "indkss"], ["WICS", '1028', "indks2"],
            ["WICS", '2002', "indkql"], ["WICS", '2003', "indkqm"], ["WICS", '2203', "indkq1"],
            ["WI26", '', "secful"],
            ["WI26", '1002', "secksl"], ["WI26", '1003', "secksm"], ["WI26", '1004', "seckss"], ["WI26", '1028', "secks2"],
            ["WI26", '2002', "seckql"], ["WI26", '2003', "seckqm"], ["WI26", '2203', "seckq1"],
            ["ETF", '', "etfful"], ["THEME", '', "thmful"]
        ]

        for kind, code, var in maps:
            self.__attr__(kind=kind, code=code)
            print(f"Proc... 시장 지도: {self.kind} / 거래소: {self.name} 수집 중... ")

            mframe = self.mapframe.copy()
            bframe = self.barframe.copy()
            assets = mframe['종목명'].tolist()
            self.labels[var] = mframe['종목코드'].tolist()
            self.covers[var] = covers = mframe['분류'].tolist()
            self.ids[var] = [
                asset + f'[{covers[n]}]' if asset in assets[n+1:] or asset in assets[:n] else asset
                for n, asset in enumerate(assets)
            ]
            self.bar[var] = bframe['종목코드'].tolist()
            self.datum = self.datum.append(
                other=mframe[~mframe['종목코드'].isin(self.datum['종목코드'])],
                ignore_index=True
            )
        self.datum.set_index(keys=['종목코드'], inplace=True)
        self.cover = self.datum[
            self.datum.index.isin([code for code in self.datum.index if '_' in code])
        ]['종목명'].tolist()
        return

    def convert(self):
        """
        시장 지도 데이터프레임 JavaScript 데이터 변환
        :return:
        """
        print("Proc... JavaScript 변환 중...")
        syntax = f'document.getElementsByClassName("date")[0].innerHTML="{self.date.year}년 {self.date.month}월 {self.date.day}일 종가 기준";'

        dir_file = os.path.join(__root__, 'warehouse/deploy/marketmap/js')
        if not os.path.isdir(dir_file):
            os.makedirs(dir_file)

        cnt = 1
        __js__ = os.path.join(dir_file, f"marketmap-{self.date.strftime('%Y%m%d')[2:]}-r{cnt}.js")
        while os.path.isfile(__js__):
            cnt += 1
            __js__ = os.path.join(dir_file, f"marketmap-{self.date.strftime('%Y%m%d')[2:]}-r{cnt}.js")

        for name, data in [
            ('labels', self.labels),
            ('covers', self.covers),
            ('ids', self.ids),
            ('bar', self.bar)
        ]:
            syntax += 'const %s = {\n' % name
            for var, val in data.items():
                syntax += '\t{}: {},\n'.format(var, str(val))
            syntax += '}\n'

        __frm__ = self.datum[['종목명', '종가', '시가총액', '크기',
                              'R1D', 'R1W', 'R1M', 'R3M', 'R6M', 'R1Y', 'PER', 'PBR', 'DIV',
                              'CR1D', 'CR1W', 'CR1M', 'CR3M', 'CR6M', 'CR1Y', 'CPER', 'CPBR', 'CDIV']].copy()
        __frm__.fillna('-', inplace=True)
        js = __frm__.to_json(orient='index', force_ascii=False)

        syntax += "const frm = {}\n".format(js)
        syntax += "const group_data = {}\n".format(str(self.cover))
        with codecs.open(
                filename=__js__,
                mode='w',
                encoding='utf-8') as __f__:
            __f__.write(jsmin.jsmin(syntax + self.suffix))
        return

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # frm = frame(date=datetime.today())
    # frm.__attr__(kind='THEME', code='')
    # print(frm.mapframe)

    marketmap = map2js(
        # date=datetime.today()
        date=datetime(2021, 11, 19)
    )
    marketmap.collect()
    marketmap.convert()