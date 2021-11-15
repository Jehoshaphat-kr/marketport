import pyautogui as pya
import time as t

pya.FAILSAFE = False

sz = pya.size()
sz_width  = sz.width
sz_height = sz.height

sec = 50
while True:
    pya.click(int(sz_width/2), 10)
    t.sleep(sec)
    pya.click(int(sz_width / 2), int(sz_height/2))
    t.sleep(sec)