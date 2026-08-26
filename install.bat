@echo off
setlocal
rem
rem Transcribe Tool - Windows Installation Launcher
rem
rem Double-click this file, or run it from cmd.exe / PowerShell:
rem   install.bat              Install core features
rem   install.bat --all        Install all features (recommended)
rem
rem Author: Antoine Lemor
rem

echo Transcribe Tool - Windows installer
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
