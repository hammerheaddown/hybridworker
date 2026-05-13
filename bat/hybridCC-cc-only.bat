@echo off
REM -------------------------------------------------------------------------
REM hybridCC-cc-only.bat - drag-and-drop CEA-608 captioner (LITE)
REM
REM Usage: drop a video file onto this .bat (or call: hybridCC-cc-only.bat path\to\video.mp4)
REM
REM What it does:
REM   1. Whisper-transcribes the source → .vtt sidecar
REM   2. Probes codec; if not H.264, transcodes the FLV intermediate to H.264
REM   3. Pipes through hybridCC-vod.exe to inject CEA-608 SEI NAL units
REM   4. Re-mux to .cc.mp4 with -a53cc 1 (so VLC's CC menu lights up)
REM
REM Output (next to source):
REM   <stem>.vtt        - WebVTT sidecar
REM   <stem>.cc.mp4     - captioned MP4 with CEA-608 in-band
REM
REM This is the LITE pipeline - no SCC sidecar, no QC, no proof PDF, no audit zip.
REM For the full audit packet, use hybridCC-full.bat (or send to cloud).
REM -------------------------------------------------------------------------

setlocal enabledelayedexpansion

REM -- Argument 1: full path to the video file (drag-drop or typed prompt) -
if "%~1"=="" (
  echo.
  echo No file dragged. Drag a video onto the prompt below, or paste a path:
  set /p "RAW_VIDEO=Path: "
) else (
  set "RAW_VIDEO=%~1"
)
if not defined RAW_VIDEO ( echo No path provided. & pause & exit /b 1 )

REM -- Strip any wrapping quotes (drag-into-cmd inserts them automatically) -
REM IMPORTANT: no outer "..." here. cmd's set "VAR=value" parser closes on
REM the first " inside `value`, so `set "X=%X:"=%"` mangles the line. The
REM unquoted form lets the substitution complete cleanly.
set RAW_VIDEO=%RAW_VIDEO:"=%

REM -- Resolve stem/dir from RAW_VIDEO (works whether dragged or typed) ----
for %%F in ("%RAW_VIDEO%") do (
  set "INPUT=%%~F"
  set "STEM=%%~nF"
  set "DIR=%%~dpF"
)
REM %DIR% always ends in \ (from %~dpF). When passed as "%DIR%" to a child
REM process, cmd parses the trailing \" as an escaped quote and eats the
REM closing quote - gluing every flag after it into one giant argument.
REM Strip the trailing backslash to keep the quote unambiguous.
set "DIR_NOSLASH=%DIR:~0,-1%"
REM Extract parent folder name (e.g. "spots") for the VTT prefix — matches
REM Modal's `<folder>_<filename>` convention so cloud + local share naming.
for %%P in ("%DIR_NOSLASH%") do set "PARENT_FOLDER=%%~nxP"
if not defined PARENT_FOLDER set "PARENT_FOLDER=local"
set "BIN=%~dp0..\..\hybridplayout\bin"
REM VTT goes to data/captions/vtt/. Captioned MP4 replaces the source.
set "VTT_DIR=%~dp0..\..\hybridplayout\data\captions\vtt"
if not exist "%VTT_DIR%" mkdir "%VTT_DIR%" 2>nul
REM Whisper writes <stem>.vtt next to source first; we move it to vtt/ with the folder prefix.
set "VTT=%DIR%%STEM%.vtt"
set "VTT_FINAL=%VTT_DIR%\%PARENT_FOLDER%_%STEM%.vtt"
set "OUT=%DIR%%STEM%.cc.mp4"

REM -- Sanity checks --------------------------------------------------------
if not exist "%INPUT%" (
  echo [ERROR] File not found: %INPUT%
  pause & exit /b 1
)
if not exist "%BIN%\faster-whisper-xxl.exe" (
  echo [ERROR] faster-whisper-xxl.exe not found at %BIN%
  pause & exit /b 1
)
if not exist "%BIN%\ffmpeg.exe" (
  echo [ERROR] ffmpeg.exe not found at %BIN%
  pause & exit /b 1
)
if not exist "%BIN%\ffprobe.exe" (
  echo [ERROR] ffprobe.exe not found at %BIN%
  pause & exit /b 1
)
if not exist "%BIN%\hybridCC-vod.exe" (
  echo [ERROR] hybridCC-vod.exe not found at %BIN%
  echo Run hybridcc\src\build-windows.bat to compile it first.
  pause & exit /b 1
)

echo.
echo =======================================================================
echo  hybridCC-cc-only - %~nx1
echo =======================================================================
echo.

REM -- Step 1: Transcribe audio + resegment to broadcast pacing -----------
echo [1/3] Transcribing audio...
set "WHISPER_JSON=%DIR%%STEM%.json"
"%BIN%\faster-whisper-xxl.exe" "%INPUT%" ^
  --model large-v2 ^
  --output_dir "%DIR_NOSLASH%" ^
  --output_format json ^
  --word_timestamps True ^
  --language en ^
  --beep_off ^
  --vad_filter true ^
  --condition_on_previous_text false >nul 2>&1
if errorlevel 1 goto :error
REM Resegment JSON -> VTT with Modal-style pacing rules.
if exist "%BIN%\vtt_resegment.exe" (
  "%BIN%\vtt_resegment.exe" --input "%WHISPER_JSON%" --output "%VTT%" >nul 2>&1
)
REM Fallback if resegmenter missing/failed: re-run Whisper for raw VTT.
if not exist "%VTT%" (
  "%BIN%\faster-whisper-xxl.exe" "%INPUT%" ^
    --model large-v2 --output_dir "%DIR_NOSLASH%" --output_format vtt ^
    --word_timestamps True --language en --beep_off ^
    --vad_filter true --condition_on_previous_text false >nul 2>&1
)
if exist "%WHISPER_JSON%" del "%WHISPER_JSON%" 2>nul
if not exist "%VTT%" (
  echo [ERROR] Whisper completed but %VTT% not found.
  goto :error
)
echo       VTT: %VTT%
echo.

REM -- Step 2: probe source codec -------------------------------------------
echo [2/3] Probing source codec...
for /f "delims=" %%C in ('""%BIN%\ffprobe.exe" -v error -select_streams v:0 -show_entries stream^=codec_name -of default^=nw^=1:nk^=1 "%INPUT%""') do set "VCODEC=%%C"
if not defined VCODEC set "VCODEC=unknown"
echo       Codec: %VCODEC%

REM Pick first-stage encoder. FLV container only accepts H.264 - anything else
REM has to re-encode before piping through hybridCC-vod.
REM -bsf:v filter_units=remove_types=6 strips ALL SEI NALs from the source
REM bitstream before our injector runs. Without it, an H.264 source that
REM already carried CEA-608 in SEI would end up with both old + new captions
REM after -c:v copy. The re-encode branch doesn't need the filter — libx264
REM doesn't carry forward source SEI on its own.
if /i "%VCODEC%"=="h264" (
  set "FIRST_VIDEO=-c:v copy -bsf:v filter_units=remove_types=6"
  echo       H.264 source - single-encode pipeline
) else (
  set "FIRST_VIDEO=-c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
  echo       Non-H.264 source - re-encoding to H.264 for FLV intermediate
)
echo.

REM -- Step 3: SEI inject pipeline ------------------------------------------
echo [3/3] Injecting CEA-608 captions...
"%BIN%\ffmpeg.exe" -y -hide_banner -loglevel error -i "%INPUT%" %FIRST_VIDEO% -c:a aac -ac 2 -ar 44100 -f flv pipe:1 2>nul | "%BIN%\hybridCC-vod.exe" "%VTT%" 2>nul | "%BIN%\ffmpeg.exe" -y -hide_banner -loglevel error -f flv -i pipe:0 -c:v copy -c:a copy -a53cc 1 -movflags +faststart "%OUT%" 2>nul
if errorlevel 1 goto :error
if not exist "%OUT%" (
  echo [ERROR] Inject completed but %OUT% not found.
  goto :error
)

REM ---- Replace the source with the captioned version ---------------------
REM Output ends up at <stem>.mp4 (replacing original). Pipeline can't write
REM straight to the source path because ffmpeg holds it open for read; so we
REM go via <stem>.cc.mp4 then swap. If the original isn't a .mp4 (.mpg, .mov,
REM etc) the legacy file gets deleted because the new container is MP4.
set "FINAL=%DIR%%STEM%.mp4"
del "%INPUT%" 2>nul
if exist "%INPUT%" (
  echo [WARN] Could not delete original "%INPUT%"
  echo        ^(VLC or another player may be holding it^).
  echo        Captioned copy saved as %OUT% instead. Close the player and
  echo        rename it manually, or re-run after closing the player.
  set "FINAL=%OUT%"
) else (
  move /y "%OUT%" "%FINAL%" >nul
  if errorlevel 1 (
    echo [WARN] Could not rename %OUT% to %FINAL% - leaving as %OUT%.
    set "FINAL=%OUT%"
  )
)

REM Move VTT to its final folder (data/captions/vtt/).
if exist "%VTT%" (
  move /y "%VTT%" "%VTT_FINAL%" >nul 2>&1
  if errorlevel 1 ( echo       [warn] Could not move VTT to %VTT_DIR% & set "VTT_FINAL=%VTT%" )
)

echo.
echo =======================================================================
echo  DONE
echo =======================================================================
echo.
echo  Captioned MP4: %FINAL%
if exist "%VTT_FINAL%" echo  VTT sidecar:   %VTT_FINAL%
echo.
echo  Verify in VLC:
echo    1. Open the captioned MP4
echo    2. Hit Play (let it start before checking the menu)
echo    3. Subtitle ^> Sub Track ^> Closed Captions 1
echo.
pause
exit /b 0

:error
echo.
echo =======================================================================
echo  PIPELINE FAILED
echo =======================================================================
echo.
echo  Check the error messages above. Common issues:
echo    - Source file unreadable / permission denied
echo    - Source has no audio track
echo    - VLC or another player is holding the output file
echo.
pause
exit /b 1
