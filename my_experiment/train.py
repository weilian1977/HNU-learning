import time
import sys 
import signal
from threading import Thread

loop = True

def noInterrupt():
    while loop:
        print("sleep begin")
        time.sleep(5)
        print("sleep end")

th = Thread(target=noInterrupt)
th.start()
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    loop = False

print("Wait for thread exit")
th.join()
print("End...")

