import pyautogui as pya
import time as t

pya.FAILSAFE = False

sz = pya.size()
sz_width  = sz.width
sz_height = sz.height

while True:
    pya.click(int(sz_width/2), 10)
    t.sleep(10)
    pya.click(int(sz_width / 2), int(sz_height/2))
    t.sleep(10)