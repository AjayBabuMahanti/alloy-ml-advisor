# AlloyML Project - PowerShell Launch Script
# Save as: launch_app.ps1
# Usage: .\launch_app.ps1

# Set script execution policy for this session (if needed)
# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

Clear-Host
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AlloyML Project - Streamlit App" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptPath = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $scriptPath

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "⚠️  Virtual environment not found!" -ForegroundColor Yellow
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: Failed to create virtual environment" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Failed to activate virtual environment" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Check if streamlit is installed
try {
    $streamlitVersion = python -c "import streamlit; print(streamlit.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Streamlit is installed (v$streamlitVersion)" -ForegroundColor Green
    }
}
catch {
    Write-Host "⚠️  Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt --quiet
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Streamlit Application" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📱 Local URL:     http://localhost:8501" -ForegroundColor Green
Write-Host "📱 Network URL:   http://$(((Get-NetIPConfiguration | Where-Object {$_.IPv4DefaultGateway -ne $null}).IPv4Address).IPAddress):8501" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   • Press Ctrl+C to stop the app" -ForegroundColor Yellow
Write-Host "   • The app will reload when you modify app.py" -ForegroundColor Yellow
Write-Host "   • Check the terminal for any errors" -ForegroundColor Yellow
Write-Host ""

# Run the Streamlit app
streamlit run apps/app.py

# Keep terminal open if app exits
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ App exited with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Check the messages above for details." -ForegroundColor Red
}
pause
