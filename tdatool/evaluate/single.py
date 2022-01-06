import pandas as pd
from tdatool.visualize import display as stock
from datetime import datetime


def is_inc(arr, size:int=5) -> bool:
    return all([arr[i] < arr[i + 1] for i in range(-1 * size, -1, 1)])

def is_dec(arr, size:int=5) -> bool:
    return all([arr[i] > arr[i + 1] for i in range(-1 * size, -1, 1)])

class estimate(stock):
    def __init__(self, ticker: str = '005930', src: str = 'github', period: int = 5, meta = None):
        super().__init__(ticker=ticker, src=src, period=period, meta=meta)

        # Usage Frames
        self._return_ = pd.DataFrame()
        self._spectra_ = pd.DataFrame()
        return

    @staticmethod
    def est_return(date:datetime, price_block:pd.DataFrame, lim:int=20) -> pd.DataFrame:
        if len(price_block) > lim:
            raise ValueError(f'price block size overloaded > lim: {lim}')

        span = price_block[['시가', '고가', '저가', '종가']].values.flatten()
        if span[0] == 0:
            return pd.DataFrame(index=[date])

        data = {}
        returns = [round(100 * (p/span[0] - 1), 2) for p in span]
        for td in [10, 15, 20]:
            _max, _min = max(returns[:td * 4]), min(returns[:td * 4])
            data.update({f'max_return_in_{td}td': _max})
            data.update({f'min_return_in_{td}td': _min})
        return pd.DataFrame(data=data, index=[date])

    @staticmethod
    def est_bb(bb:pd.DataFrame, backtest:bool=False) -> tuple:
        """
        볼린저밴드 평가 지표(instant point)
        ** MA20 상승 판정이나 MA선 부근까지 하락한 상태는 변동성 판단 필요
        :param bb: 5TD 이상 포함 볼린저밴드 데이터프레임
        :param backtest: 과거 테스트 여부
        :return:
        """
        date = bb.index[-1]

        if datetime(2020, 3, 14) <= date <= datetime(2020, 3, 25):
            return date, 0, '강제', 1, 1, 1

        c, h, l, m, u, d, w = bb.종가, bb.고가, bb.저가, bb.기준선, bb.상한선, bb.하한선, bb.밴드폭

        # 상태 판단
        is_upper, is_lower = (c[-1] > u[-1]), (c[-1] < d[-1])
        is_rise, is_fall = is_inc(m), is_dec(m)
        is_widen = is_inc(w)
        is_rise_fast = is_inc(c)

        # 종가 위치에 따른 저항선 대비 상승분
        _c_ = lambda goal, here: 100 * (goal[-1] / here[-1] - 1)
        capa = _c_(c, u) if is_upper else _c_(h, u) if h[-1] > u[-1] else _c_(u, c) if c[-1] > m[-1] else _c_(m, c)

        # 상승/하강 속도 펙터
        dy_curr, dy_base = (m[-1] - m[-3]), (m[-1] - m[-5])
        k_vel = 2 * dy_curr / dy_base if is_rise else 0.5 * dy_base / dy_curr if is_fall else 1

        # 폭 팩터
        r_acc = max([2 * (w[-1] - w[i]) / (w[-1] - w[-5]) for i in range(-4, -1, 1)])
        k_width = (r_acc if is_rise else r_acc ** -1 if is_fall else 1) if is_widen and r_acc > 1 else 1

        # 기준선 대비 실 주가 괴리 보상
        refl = max([100 * (c[-1] / c[i] - 1) for i in range(-5, -1, 1)])
        capa = refl - capa if (refl > capa and is_fall) or (is_fall and is_rise_fast) else capa

        score = k_width * k_vel * capa
        if backtest:
            return date, score, '상승' if is_rise else '하락' if is_fall else '보합', capa, k_vel, k_width
        else:
            return score

    @staticmethod
    def est_overtrade(frm:pd.DataFrame, backtest:bool=False) -> tuple:
        """

        :param frm:
        :param backtest:
        :return:
        """

        date = frm.index[-1]
        o, h, l, c = frm.시가, frm.고가, frm.저가, frm.종가
        rsi, s_rsi, s_rsi_sig, cci = frm.RSI, frm['STOCH-RSI'], frm['STOCH-RSI-Sig'], frm.CCI

        price_steps = [1 if c[i] < c[i + 1] else 0 for i in range(-10, -1, 1)]
        is_price_rise = all([c[i] < c[-1] for i in range(-10, -1, 1)]) and sum(price_steps) >= 5
        is_price_fall = all([c[i] > c[-1] for i in range(-10, -1, 1)]) or sum(price_steps) < 5

        cci_steps = [1 if cci[i] < cci[i+1] else 0 for i in range(-10, -1, 1)]
        is_cci_rise = all([cci[i] < cci[-1] for i in range(-10, -1, 1)]) and sum(cci_steps) >= 5
        is_cci_fall = all([cci[i] > cci[-1] for i in range(-10, -1, 1)]) or sum(cci_steps) < 5

        # CAPACITY
        if cci[-1] > 200:
            capa = cci[-1] - 200
        elif cci[-1] > 100:
            capa = 200 - cci[-1]
        elif cci[-1] > 0:
            capa = 100 - cci[-1]
        elif cci[-1] > -100:
            capa = 0 - cci[-1]
        else:
            capa = -100 - cci[-1]

        # Velocity Factor
        numer = cci[-1] - cci[-2]
        denom = max([(cci[-1] - cci[i]) / (abs(i) - 1) for i in [-6, -5, -4]])
        # denom = (cci[-1] - cci[-5]) / 4
        slope = numer / denom
        if slope > 1:
            acc = slope
            dec = slope ** -1
        else:
            acc = slope ** -1
            dec = slope

        k_vel = acc if is_cci_rise and is_price_rise else dec if is_cci_fall or is_price_fall else 1


        # score = capa if is_cci_rise and is_price_rise else 0.5 * capa if is_cci_fall or is_price_fall else 0
        score = k_vel * capa

        if backtest:
            return date, score, '상승' if is_cci_rise and is_price_rise else '하락' if is_cci_fall else '보합'
        else:
            return score

    @property
    def historical_return(self) -> pd.DataFrame:
        """
        백테스트 수익률 데이터프레임
        :return:
        """
        if self._return_.empty:
            p = self.price_ori[['시가', '고가', '저가', '종가']].copy()
            objs = [self.est_return(date=d, price_block=p[i + 1: i + 21]) for i, d in enumerate(p.index[:-1])]
            self._return_ = pd.concat(objs=objs, axis=0)
        return self._return_

    @property
    def historical_bb_estimate(self) -> pd.DataFrame:
        """
        볼린저밴드 평가 백테스트
        :return:
        """
        bb = pd.concat([self.price, self.bollinger], axis=1)
        data = [self.est_bb(bb=bb[i - 5 : i], backtest=True) for i in range(5, len(bb))]
        score = pd.DataFrame(data=data, columns=['날짜', '점수', '판정', '기본점', '가속팩터', '폭팩터']).set_index(keys='날짜')
        return bb.join(score, how='left')

    @property
    def historical_trading_estimate(self) -> pd.DataFrame:
        """
        과매매 평가 백테스트
        :return:
        """
        frm = pd.concat([self.price, self.rsi, self.stoch_rsi, self.cci], axis=1)
        data = [self.est_overtrade(frm=frm[i - 10 : i], backtest=True) for i in range(10, len(frm))]
        score = pd.DataFrame(data=data, columns=['날짜', '점수', '판정']).set_index(keys='날짜')
        return frm.join(score, how='left')

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
    title = 'T03-보상-상승시'
    tickers = [
        # '005930', # 삼성전자
        # '000660', # SK하이닉스
        '006400', # 삼성SDI
        # '066570', # LG전자
        # '018260', # 삼성에스디에스
        # '009150', # 삼성전기
        # '005380', # 현대차
        # '035420', # NAVER
        # '035720', # 카카오
        # '051900', # LG생활건강
        # '090430', # 아모레퍼시픽
        # '105560', # KB금융
        # '055550', # 신한지주
        # '051910', # LG화학
        # '005490', # POSCO
        # '207940', # 삼성바이오로직스
        # '068270', # 셀트리온
        # '096770', # SK이노베이션
    ]

    objs = []
    for ticker in tickers:
        ev = estimate(ticker=ticker, src='local', period=10)
        print(f"{ev.name}({ev.ticker})")
        ach = ev.historical_return
        # est = ev.historical_bb_estimate
        est = ev.historical_trading_estimate
        _frm = pd.concat([est, ach], axis=1)
        _frm.index.name = '날짜'
        _frm.reset_index(level=0, inplace=True)
        _frm['종목'] = ev.name
        objs.append(_frm)
    frm = pd.concat(objs=objs, axis=0)

    # frm.to_csv(rf'text_{title}.csv', encoding='euc-kr', index=False)


    # scale = ['#F63538', '#BF4045', '#8B444E', '#414554', '#35764E', '#2F9E4F', '#30CC5A']
    key = 'max_return_in_20td'
    scale = ['#414554', '#35764E', '#2F9E4F', '#30CC5A']
    bins = [0.0, 1.0, 3.0, 5.0]
    frm['color'] = pd.cut(frm[key], bins=bins + [frm[key].max()], labels=scale, right=True)

    score = frm['점수'].copy().sort_values(ascending=False)
    s_avg, s_std = score.mean(), score.std()
    score_lim_by_class = [score.tolist()[int(r * len(frm))] for r in [0.01, 0.04, 0.11, 0.21]]

    comment = f"데이터 개수     : {len(frm)}<br>"
    comment += f"전체 평균 score : {s_avg}<br>"
    comment += f"       표준편차 : {s_std}<br>"
    comment += f"전체 평균수익률 : {round(frm[key].mean(), 2)} %<br>"
    for lim in score_lim_by_class:
        # lim = (s_avg + s * s_std) if _ == 1 else s * 5

        _frm = frm[frm['점수'] > lim].copy()
        avg = _frm[key].mean()
        n_ovr = len(_frm[_frm[key] > 5])
        comment += f'score > {round(lim, 2)} 비율        : {round(100 * len(_frm) / len(frm), 2)} %<br>'
        comment += f'score > {round(lim, 2)} 평균 수익률 : {round(avg, 2)} %<br>'
        comment += f'score > {round(lim, 2)} 달성률      : {round(100 * n_ovr/len(_frm), 2)} %<br>'
        comment += ('-' * 50 + '<br>')


    import plotly.graph_objects as go
    import plotly.offline as of
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frm['점수'], y=frm[key],
        mode='markers', marker=dict(color=frm.color),
        meta=[f'{d.year}/{d.month}/{d.day}' for d in frm['날짜']], text=frm['판정'], customdata=frm['종목'],
        hovertemplate='%{customdata}<br>날짜: %{meta}<br>점수: %{x}<br>수익률: %{y}%<br>판정: %{text}<extra></extra>'
    ))
    for lim in score_lim_by_class:
        fig.add_vline(x=lim, line=dict(dash='dot', color='black'))

    fig.update_layout(
        title=title,
        xaxis=dict(title_text='score'),
        yaxis=dict(title_text=key),
        annotations=[
            dict(
                text=comment,
                showarrow=False,
                xref="paper", yref="paper", x=1.0, y=1.0, align="left"
            )
        ],
    )
    of.plot(fig, filename='./test.html', auto_open=False)