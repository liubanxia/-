@echo off
setlocal
where bash >nul 2>nul
if errorlevel 1 (
  echo ERROR: Git Bash not found in PATH.
  echo Open Git Bash and run scripts/netcafe_bootstrap.sh directly.
  exit /b 1
)

bash "%~dp0netcafe_bootstrap.sh" %*
exit /b %errorlevel%
