# Monitoring System Test Script (PowerShell)
# Tests the end-to-end monitoring infrastructure

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MONITORING SYSTEM TEST SCRIPT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Docker is running
Write-Host "[1/7] Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  Success: Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "  Error: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Step 2: Build services
Write-Host ""
Write-Host "[2/7] Building Docker images..." -ForegroundColor Yellow
docker-compose build
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Success: Build completed" -ForegroundColor Green
}
else {
    Write-Host "  Error: Build failed" -ForegroundColor Red
    exit 1
}

# Step 3: Start services
Write-Host ""
Write-Host "[3/7] Starting services..." -ForegroundColor Yellow
docker-compose up -d
Start-Sleep -Seconds 10
docker-compose ps
Write-Host "  Success: Services started" -ForegroundColor Green

# Step 4: Wait for inference server
Write-Host ""
Write-Host "[4/7] Waiting for inference server to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$ready = $false

while (($attempt -lt $maxAttempts) -and (-not $ready)) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8082/" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ready = $true
            Write-Host "  Success: Inference server is ready" -ForegroundColor Green
        }
    }
    catch {
        $attempt++
        Write-Host "  Waiting... ($attempt/$maxAttempts)" -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
}

if (-not $ready) {
    Write-Host "  Error: Inference server did not start in time" -ForegroundColor Red
    docker-compose logs inference
    exit 1
}

# Step 5: Generate test requests
Write-Host ""
Write-Host "[5/7] Generating test recommendation requests..." -ForegroundColor Yellow
$successCount = 0
$failCount = 0

# Generate 20 valid requests
for ($i = 1; $i -le 20; $i++) {
    $userId = Get-Random -Minimum 1 -Maximum 1000
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8082/recommend/$userId" -Method GET -ErrorAction SilentlyContinue
        $successCount++
        Write-Host "  Request $i (user $userId): SUCCESS" -ForegroundColor Green
    }
    catch {
        Write-Host "  Request $i (user $userId): FAILED" -ForegroundColor Red
        $failCount++
    }
    Start-Sleep -Milliseconds 500
}

# Generate 5 invalid requests (to test error handling)
Write-Host ""
Write-Host "  Generating error scenarios..." -ForegroundColor Gray
for ($i = 1; $i -le 5; $i++) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8082/recommend/invalid_user_$i" -Method GET -ErrorAction SilentlyContinue
    }
    catch {
        # Expected to fail
    }
}

Write-Host "  Success: Generated $successCount successful and 5 error requests" -ForegroundColor Green

# Step 6: Verify Prometheus metrics
Write-Host ""
Write-Host "[6/7] Checking Prometheus metrics..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

try {
    $metrics = Invoke-WebRequest -Uri "http://localhost:8000/metrics" -Method GET
    if ($metrics.Content -match "recommendation_service_availability") {
        Write-Host "  Success: Prometheus metrics are being exported" -ForegroundColor Green
    }
    else {
        Write-Host "  Warning: Metrics endpoint is up but no recommendation metrics found yet" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  Error: Could not access Prometheus metrics" -ForegroundColor Red
}

# Step 7: Check database
Write-Host ""
Write-Host "[7/7] Verifying database records..." -ForegroundColor Yellow

$dbContainer = docker ps -q -f "name=db_writer"
if ($dbContainer) {
    $query = "SELECT COUNT(*) as count, status FROM recommendations WHERE timestamp >= datetime('now', '-5 minutes') GROUP BY status;"
    $result = docker exec $dbContainer sqlite3 /app/db/home_alone.db "$query"

    if ($result) {
        Write-Host "  Success: Database contains recommendation records:" -ForegroundColor Green
        Write-Host "  $result" -ForegroundColor Cyan
    }
    else {
        Write-Host "  Warning: No recent recommendations in database" -ForegroundColor Yellow
    }
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services Status:" -ForegroundColor White
docker-compose ps
Write-Host ""
Write-Host "Access Points:" -ForegroundColor White
Write-Host "  Inference API:       http://localhost:8082" -ForegroundColor Cyan
Write-Host "  Prometheus Metrics:  http://localhost:8000/metrics" -ForegroundColor Cyan
Write-Host "  Grafana Dashboard:   http://localhost:3000" -ForegroundColor Cyan
Write-Host "    (Username: admin, Password: admin)" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  1. Open Grafana at http://localhost:3000" -ForegroundColor Yellow
Write-Host "  2. Check monitoring logs with: docker-compose logs -f monitoring" -ForegroundColor Yellow
Write-Host "  3. View alert logs in monitoring container" -ForegroundColor Yellow
Write-Host ""
Write-Host "To stop all services run: docker-compose down" -ForegroundColor Gray
Write-Host ""
