import requests, time, threading

URL = "https://api.regolo.ai/v1/chat/completions"
HEADERS = {"Authorization": "Bearer LA_TUA_KEY"}

def call(i):
    start = time.time()
    r = requests.post(URL, json={
        "model": "Llama-3.3-70B-Instruct",
        "messages": [{"role": "user", "content": f"ciao {i}"}]
    }, headers=HEADERS)
    print(i, time.time() - start)

threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
[t.start() for t in threads]
