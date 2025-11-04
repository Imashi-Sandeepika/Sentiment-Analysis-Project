# Git Setup and Commit Script for Sentiment Analysis Project
# Run this script after installing Git from https://git-scm.com/download/win

Write-Host "Setting up Git repository..." -ForegroundColor Green

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Initialize git if not already initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing Git repository..." -ForegroundColor Cyan
    git init
}

# Add all files
Write-Host "Adding files to staging..." -ForegroundColor Cyan
git add .

# Commit with message
Write-Host "Creating commit..." -ForegroundColor Cyan
git commit -m "feat: Add sentiment analysis model and prediction pipeline

- Implement model building notebook (02.model_building.ipynb)
- Create prediction pipeline (03.Prediction_pipeline.ipynb)
- Add trained model artifact (model.pickle)
- Configure project dependencies in requirement.txt
- Add .gitignore for Python/Jupyter environment
- Include static assets and artifacts directory"

Write-Host "`nCommit successful!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub: https://github.com/new"
Write-Host "2. Run these commands to push:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/Sentiment-Analysis-Project.git"
Write-Host "   git branch -M main"
Write-Host "   git push -u origin main"
