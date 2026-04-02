import time

min_interval = 15  # seconds (4 api calls/min)
last_vt_call = 0.0

def rate_limit():
    global last_vt_call
    elapsed = time.time() - last_vt_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    last_vt_call = time.time()