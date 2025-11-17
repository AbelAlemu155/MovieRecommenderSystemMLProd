#!/usr/bin/env python3
"""
Generate realistic traffic for monitoring dashboard
Makes random recommendation requests - the system naturally routes to SVD or Fallback
"""

import requests
import random
import time

print("Generating realistic recommendation traffic...")
print("System will automatically route to appropriate model (SVD/Fallback)")
print()

success_count = 0
fail_count = 0

# Generate random requests - system decides which model to use
num_requests = 30
for i in range(1, num_requests + 1):
    # Random user ID - could be in trainset (SVD) or not (Fallback)
    user_id = random.randint(1, 100000)

    try:
        response = requests.get(f"http://localhost:8082/recommend/{user_id}", timeout=5)
        if response.status_code == 200:
            success_count += 1
            print(f"  Request {i:2d} (user {user_id:6d}): SUCCESS")
        else:
            fail_count += 1
            print(f"  Request {i:2d} (user {user_id:6d}): FAILED (HTTP {response.status_code})")
    except Exception as e:
        fail_count += 1
        print(f"  Request {i:2d} (user {user_id:6d}): ERROR ({str(e)[:40]})")

    time.sleep(0.3)

print()
print("=" * 60)
print("Traffic generation completed!")
print(f"  Successful requests: {success_count}")
print(f"  Failed requests: {fail_count}")
print()
print("The system automatically distributed requests across models:")
print("  - SVD model (for users in training set)")
print("  - Fallback model (for new/unknown users)")
print()
print("Wait 30 seconds for metrics to update, then check:")
print("  - Grafana dashboard: http://localhost:3000")
print("  - Prometheus metrics: http://localhost:8000/metrics")
