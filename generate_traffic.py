#!/usr/bin/env python3
"""Generate test traffic for monitoring dashboard"""

import requests
import random
import time

print("Generating test recommendation requests...")
print()

success_count = 0
fail_count = 0

# Generate 25 valid requests
for i in range(1, 26):
    user_id = random.randint(1, 1000)
    try:
        response = requests.get(f"http://localhost:8082/recommend/{user_id}", timeout=5)
        if response.status_code == 200:
            success_count += 1
            print(f"  Request {i} (user {user_id}): SUCCESS")
        else:
            fail_count += 1
            print(f"  Request {i} (user {user_id}): FAILED (HTTP {response.status_code})")
    except Exception as e:
        fail_count += 1
        print(f"  Request {i} (user {user_id}): FAILED ({str(e)})")

    time.sleep(0.5)

print()
print("Test traffic generation completed!")
print(f"  Successful requests: {success_count}")
print(f"  Failed requests: {fail_count}")
print()
print("Wait 30 seconds for Prometheus to scrape metrics, then refresh your Grafana dashboard at http://localhost:3000")
