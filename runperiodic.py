import docker, time
from datetime import datetime


if __name__ == "__main__":
    today = datetime.today()
    # today = datetime(2021, 10, 15)

    ''' GEN META-DATA '''
    docker.basis(date=today).update()
    time.sleep(1)

    ''' UPDATE PRICE '''
    docker.price(date=today).update(debug=False)
    time.sleep(2)

    ''' UPDATE GROUP '''
    docker.group().update_group(kind='wics')
    docker.group().update_group(kind='wi26')

    ''' GEN MARKET-DATA '''
    market = docker.interface(date=today)
    market.update_percentage()
    market.update_multiple()
    market.save()
    time.sleep(2)

    ''' MARKET MAP '''
    maps = docker.marketmap(date=today)
    maps.collect()
    maps.convert()