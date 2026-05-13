@echo off
REM ---------------------------------------------------------------------
REM transcribe-only.bat - drag-and-drop audio/video transcription
REM
REM Output: <stem>.vtt next to the source file. Nothing else.
REM No QC, no SEI inject, no audit zip - just a raw Whisper VTT.
REM Useful for: debugging, sanity-checking transcription quality before
REM running the full pipeline, or just getting a sidecar caption track.
REM ---------------------------------------------------------------------

setlocal enabledelayedexpansion

REM -- Argument 1: full path to the video file (drag-drop or typed prompt) --
if "%~1"=="" (
  echo.
  echo No file dragged. Drag a video onto the prompt below, or paste a path:
  set /p "RAW_VIDEO=Path: "
) else (
  set "RAW_VIDEO=%~1"
)
if not defined RAW_VIDEO ( echo No path provided. & pause & exit /b 1 )

REM -- Strip any wrapping quotes (drag-into-cmd inserts them automatically) -
set RAW_VIDEO=%RAW_VIDEO:"=%

REM -- Resolve stem/dir --
for %%F in ("%RAW_VIDEO%") do (
  set "INPUT=%%~F"
  set "STEM=%%~nF"
  set "DIR=%%~dpF"
  set "SRC_NAME=%%~nxF"
)
set "DIR_NOSLASH=%DIR:~0,-1%"
set "BIN=%~dp0..\..\hybridplayout\bin"
set "VTT=%DIR%%STEM%.vtt"

if not exist "%INPUT%" ( echo [ERROR] File not found: %INPUT% & pause & exit /b 1 )
if not exist "%BIN%\faster-whisper-xxl.exe" ( echo [ERROR] transcription tool not found at %BIN% & pause & exit /b 1 )

echo.
echo =======================================================================
echo  Transcribe - %SRC_NAME%
echo =======================================================================
echo.
echo  Output will be: %VTT%
echo.

"%BIN%\faster-whisper-xxl.exe" "%INPUT%" ^
  --model large-v2 ^
  --output_dir "%DIR_NOSLASH%" ^
  --output_format vtt ^
  --word_timestamps True ^
  --language en ^
  --beep_off ^
  --vad_filter true ^
  --condition_on_previous_text false

if errorlevel 1 (
  echo.
  echo [ERROR] Transcribe failed.
  pause & exit /b 1
)

if not exist "%VTT%" (
  echo.
  echo [ERROR] Transcribe completed but %VTT% not found.
  pause & exit /b 1
)

echo.
echo =======================================================================
echo  DONE
echo =======================================================================
echo  VTT: %VTT%
echo.
pause
exit /b 0
