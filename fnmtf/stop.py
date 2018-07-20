import numpy as np

def check_stop(E, history, stop='p6-10'):
    prev = history[-2]
    now = history[-1]
    
    if E is not None:
        now = E.fetch()[0,0]
    
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
    
    if stop == 'd3':
        if 10**(-3) * np.abs(now) - np.abs(prev - now) > 0:
            return True
    
    if stop == 'd4':
        if 10**(-4) * np.abs(now) - np.abs(prev - now) > 0:
            return True
    
    if stop == 'd5':
        if 10**(-5) * np.abs(now) - np.abs(prev - now) > 0:
            return True
    
    if stop == 'd6':
        if 10**(-6) * np.abs(now) - np.abs(prev - now) > 0:
            return True
            
    if stop == 'd7':
        if 10**(-7) * np.abs(now) - np.abs(prev - now) > 0:
            return True

    if stop == 'p5-5':
        if check_lastN(history, 10**(-5), 6):
            return True
    
    if stop == 'p5-10':
        if check_lastN(history, 10**(-5), 11):
            return True
    
    if stop == 'p6-5':
        if check_lastN(history, 10**(-6), 6):
            return True
    
    if stop == 'p6-10':
        if check_lastN(history, 10**(-6), 11):
            return True
    
    if stop == 'p7-5':
        if check_lastN(history, 10**(-7), 6):
            return True
    
    if stop == 'p7-10':
        if check_lastN(history, 10**(-7), 11):
            return True
    
    prev = now
    return False
