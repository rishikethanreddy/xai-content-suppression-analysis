import requests
import json
payload = {
    'text': '2.4K 11 babunuvubtechah A female engineering student in Bachupally was allegedly drugged and sexually assaulted by a fellow student for nearly year: The case came to light after she reportedly attempted suicide: The accused allegedly threatened to leak private photos Family claims college staff ignored earlier complaints_ Police investigation is ongoing:',
    'hashtags': [],
    'likes': 2400,
    'watch_time': 50.0
}
r = requests.post('http://localhost:8000/predict-post', json=payload)
print(json.dumps(r.json(), indent=2))
