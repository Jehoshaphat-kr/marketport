import docker, time
from datetime import datetime


if __name__ == "__main__":
    today = datetime.today()
    # today = datetime(2021, 10, 13)

    '''
    GEN META-DATA
    '''
    docker.basis(date=today).update()
    time.sleep(1)

    '''
    UPDATE PRICE
    '''
    docker.price(date=today).update(debug=False)
    time.sleep(2)

    '''
    GEN MARKET-DATA
    '''
    market = docker.interface(date=today)
    market.update_percentage()
    market.update_multiple()
    market.save()
    time.sleep(1)


    '''
    FETCH INDEX
    '''
    docker.index(date=today).update(debug=False)
    time.sleep(1)

    '''
    UPDATE ETF DEPOSIT
    '''
    docker.deposit(date=today).update()
