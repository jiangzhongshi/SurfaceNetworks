import datetime
import time
import torch
def time_string():
    time = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(-1, 72000))).strftime("%b-%d-%I:%M%p")
    return f'NewYork {time}'

def cutimer(log=None):
    if log is None:
        torch.cuda.synchronize()
        timer.time0 = time.time()
    else:
        torch.cuda.synchronize()
        end = time.time()
        print(f'{log}: {end-timer.time0}')

def timer(log=None):
    if log is None:
        timer.time0 = time.time()
    else:
        end = time.time()
        print(f'{log}: {end-timer.time0}')


