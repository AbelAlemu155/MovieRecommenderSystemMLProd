#!/bin/bash

echo "Starting zero-downtime test..."
echo "1. Starting background request loop..."

# Background loop
(while true; do
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8082/recommend/42
  sleep 1
done) > requests.log &
REQUEST_PID=$!

echo "2. Waiting 5 seconds for baseline..."
sleep 5

echo "3. Triggering model reload..."
docker exec home-alone-inference-1-1 touch /app/models/svd_model.dump

echo "4. Monitoring for 120 seconds (enough for all replicas to reload)..."
sleep 120

echo "5. Stopping test..."
kill $REQUEST_PID

echo "6. Results:"
total=$(wc -l < requests.log)
success=$(grep -c "200" requests.log)
failures=$((total - success))

echo "Total requests: $total"
echo "Successful: $success"
echo "Failed: $failures"

if [ $failures -eq 0 ]; then
  echo "ZERO DOWNTIME CONFIRMED!"
else
  echo "DOWNTIME DETECTED - $failures requests failed"
fi

rm requests.log
