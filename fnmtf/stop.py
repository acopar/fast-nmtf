import numpy as np

def check_lastN(history, EPS, N, I):
    lastN = history[I-N-1:I]
    condition = 0
    for i in range(1,N+1):
        prev = lastN[i-1]
        now = lastN[i]
        
        if EPS * np.abs(now) - np.abs(prev - now) > 0:
            condition = 1
        elif now > prev:
            condition = 1
        else:
            condition = 0
            break
    
    return condition
    
def score_history(hist, stop='d', epsilon=7, wait=0):
    if stop == 'd':
        i = len(hist)-1
        if 10**(-epsilon) * np.abs(hist[i]) - np.abs(hist[i-1] - hist[i]) > 0:
            return i
    elif stop == 'p5':
        i = len(hist)
        if len(hist) > 5:
            EPS = 10**(-epsilon)
            if check_lastN(hist, EPS, 5, i):
                return i
    elif stop == 'p10':
        i = len(hist)
        if len(hist) > 10:
            EPS = 10**(-epsilon)
            if check_lastN(hist, EPS, 10, i):
                return i
    return -1

def score_history2(hist, stop='d', epsilon=7, wait=0):
    if stop == 'd':
        i = len(hist)-1
        if 10**(-epsilon) * np.abs(hist[i]) - np.abs(hist[i-1] - hist[i]) > 0:
            return i
    elif stop == 'p5':
        for i in range(6+wait,len(hist)+1):
            if len(hist) > 5:
                EPS = 10**(-epsilon)
                if check_lastN(hist, EPS, 5, i):
                    return i
    elif stop == 'p10':
        #i = len(hist)
        for i in range(11+wait,len(hist)+1):
            EPS = 10**(-epsilon)
            cl = check_lastN(hist, EPS, 10, i)
            if cl == 1:
                return i
            elif cl == 2:
                return -2
    return -1

def check_stop(history, stop='p6-10'):
    prev = history[-2]
    now = history[-1]
    
    if stop == 'e4':
        if np.abs(prev - now) < 10**(-4):
            return True
    
    if stop == 'e5':
        if np.abs(prev - now) < 10**(-5):
            return True
    
    if stop == 'e6':
        if np.abs(prev - now) < 10**(-6):
            return True
            
    if stop == 'e7':
        if np.abs(prev - now) < 10**(-7):
            return True
