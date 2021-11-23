import tdatool as tt

if __name__ == "__main__":
    technical = tt.TimeSeries(
        ticker='011790',
        data_src='offline'
    )
    print(f'{technical.name}({technical.ticker}) 기술 분석')
    technical.s_price(show=False, save=True)
    technical.s_trend(show=False, save=True)


    fundamental = tt.Finances(
        ticker='011790'
    )