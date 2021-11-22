

class Equity:

    def __init__(
        self,
        ticker: str = '005930',
        data_src: str = 'online',
        filter_key: str = '저가',
        filter_win: list = [5, 10, 20, 60, 120],
    ):
        return

if __name__ == "__main__":
    eq = Equity(ticker='005930')
