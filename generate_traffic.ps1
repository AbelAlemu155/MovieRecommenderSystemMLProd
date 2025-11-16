# Generate test traffic for monitoring dashboard
Write-Host "Generating test recommendation requests..." -ForegroundColor Cyan
Write-Host ""

$successCount = 0
$failCount = 0

# Generate 25 valid requests using curl (to avoid PowerShell Invoke-WebRequest issues with Flask debug mode)
for ($i = 1; $i -le 25; $i++) {
    $userId = Get-Random -Minimum 1 -Maximum 1000
    $result = curl -s -w "%{http_code}" -o nul "http://localhost:8082/recommend/$userId"

    if ($result -eq "200") {
        $successCount++
        Write-Host "  Request $i (user $userId): SUCCESS" -ForegroundColor Green
    }
    else {
        Write-Host "  Request $i (user $userId): FAILED (HTTP $result)" -ForegroundColor Red
        $failCount++
    }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "Test traffic generation completed!" -ForegroundColor Cyan
Write-Host "  Successful requests: $successCount" -ForegroundColor Green
Write-Host "  Failed requests: $failCount" -ForegroundColor Red
Write-Host ""
Write-Host "Refresh your Grafana dashboard at http://localhost:3000" -ForegroundColor Yellow
