@echo off
REM -------------------------------------------------------------------------
REM hybridCC-full.bat - drag-and-drop full audit pipeline
REM
REM Usage: drop a video file onto this .bat
REM
REM What it does:
REM   1. Whisper-transcribes the source → .vtt
REM   2. Runs vtt_to_scc QC → cleaned .vtt + cleaned .srt + .scc + qc.json
REM      (REQUIRES bin\vtt_to_scc.exe - built via PyInstaller. If missing,
REM       the script falls back to LITE behavior with a warning.)
REM   3. Probes codec; transcodes FLV intermediate to H.264 if needed
REM   4. Pipes through hybridCC-vod.exe to inject CEA-608 SEI
REM   5. Re-mux to .cc.mp4 with -a53cc 1
REM   6. Renders proof-report.html → proof.pdf via Edge headless (msedge)
REM   7. Bundles .scc + .srt + .pdf into <stem>.audit.zip
REM
REM Output (next to source):
REM   <stem>.cc.mp4           - captioned MP4
REM   <stem>.vtt              - cleaned VTT
REM   <stem>.audit.zip        - slim audit packet (PDF + SCC + SRT)
REM
REM Pre-existing tools needed in hybridplayout\bin\:
REM   - faster-whisper-xxl.exe (already there)
REM   - ffmpeg.exe / ffprobe.exe (already there)
REM   - hybridCC-vod.exe (built via hybridcc\src\build-windows.bat)
REM   - vtt_to_scc.exe (TODO: PyInstaller from hybridcc\vtt_to_scc.py)
REM
REM Edge headless: uses C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
REM (already on every Windows 10/11 machine - no install needed)
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
REM No outer "..." - see hybridCC-cc-only.bat for the explanation.
set RAW_VIDEO=%RAW_VIDEO:"=%

REM -- Resolve stem/dir from RAW_VIDEO (works whether dragged or typed) ----
for %%F in ("%RAW_VIDEO%") do (
  set "INPUT=%%~F"
  set "STEM=%%~nF"
  set "DIR=%%~dpF"
  set "SRC_NAME=%%~nxF"
)
REM Strip trailing \ so passing as "%DIR_NOSLASH%" doesn't escape the quote.
set "DIR_NOSLASH=%DIR:~0,-1%"
REM Extract the parent folder name (e.g. "spots" from .../media/spots/) so
REM audit zip + VTT filenames carry it as a prefix — matches Modal's
REM `<folder>_<filename>.audit.zip` convention so cloud + local share a
REM single naming scheme in data/captions/.
for %%P in ("%DIR_NOSLASH%") do set "PARENT_FOLDER=%%~nxP"
if not defined PARENT_FOLDER set "PARENT_FOLDER=local"
set "BIN=%~dp0..\..\hybridplayout\bin"
REM Split layout under data/captions/:
REM   data/captions/audit/<stem>.audit.zip   - durable bundle (PDF + SCC + SRT)
REM   data/captions/vtt/<stem>.vtt           - WebVTT sidecar
REM   <source folder>/<stem>.mp4             - captioned video (overwrites source)
REM Intermediates (qc.json, scc, srt, proof.html/pdf) live in audit/ during
REM processing then get deleted after the audit zip is built.
set "AUDIT_DIR=%~dp0..\..\hybridplayout\data\captions\audit"
set "VTT_DIR=%~dp0..\..\hybridplayout\data\captions\vtt"
if not exist "%AUDIT_DIR%" mkdir "%AUDIT_DIR%" 2>nul
if not exist "%VTT_DIR%"   mkdir "%VTT_DIR%"   2>nul
REM Intermediate paths use the bare stem (auto-deleted at end of pipeline).
set "WHISPER_JSON=%AUDIT_DIR%\%STEM%.json"
set "VTT=%AUDIT_DIR%\%STEM%.vtt"
set "VTT_CLEAN=%AUDIT_DIR%\%STEM%.cleaned.vtt"
set "SRT=%AUDIT_DIR%\%STEM%.srt"
set "SCC=%AUDIT_DIR%\%STEM%.scc"
set "QC=%AUDIT_DIR%\%STEM%.qc.json"
set "PROOF_HTML=%AUDIT_DIR%\%STEM%.proof.html"
set "PROOF_PDF=%AUDIT_DIR%\%STEM%.proof.pdf"
REM Durable artifacts use the Modal-style folder-prefixed name.
REM Audit zip name mirrors what cloud captionUnpack writes:
REM   sanitize(mediaId='spots/test_mpeg2.mpg') = 'spots_test_mpeg2.mpg'
REM   final filename = 'spots_test_mpeg2.mpg.audit.zip'
set "VTT_FINAL=%VTT_DIR%\%PARENT_FOLDER%_%STEM%.vtt"
set "AUDIT_ZIP=%AUDIT_DIR%\%PARENT_FOLDER%_%SRC_NAME%.audit.zip"
REM .cc.mp4 stays next to source — needed because ffmpeg can't write to source while reading from it.
set "OUT=%DIR%%STEM%.cc.mp4"
set "MSEDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
REM AUDIT_DIR_NOSLASH for child --output_dir (avoid trailing-quote escape).
set "AUDIT_DIR_NOSLASH=%AUDIT_DIR%"

REM -- Sanity checks --------------------------------------------------------
if not exist "%INPUT%" ( echo [ERROR] File not found: %INPUT% & pause & exit /b 1 )
if not exist "%BIN%\faster-whisper-xxl.exe" ( echo [ERROR] missing faster-whisper-xxl.exe & pause & exit /b 1 )
if not exist "%BIN%\ffmpeg.exe" ( echo [ERROR] missing ffmpeg.exe & pause & exit /b 1 )
if not exist "%BIN%\ffprobe.exe" ( echo [ERROR] missing ffprobe.exe & pause & exit /b 1 )
if not exist "%BIN%\hybridCC-vod.exe" (
  echo [ERROR] hybridCC-vod.exe not found at %BIN%
  echo Run hybridcc\src\build-windows.bat first.
  pause & exit /b 1
)

set "HAS_VTT2SCC=0"
if exist "%BIN%\vtt_to_scc.exe" set "HAS_VTT2SCC=1"

set "HAS_EDGE=0"
if exist "%MSEDGE%" set "HAS_EDGE=1"

echo.
echo =======================================================================
echo  HybridCC - %SRC_NAME%
echo =======================================================================
echo.

REM -- Step 1: Transcribe audio (JSON output for resegmenter) --------------
echo [1/6] Transcribing audio...
"%BIN%\faster-whisper-xxl.exe" "%INPUT%" ^
  --model large-v2 --output_dir "%AUDIT_DIR_NOSLASH%" --output_format json ^
  --word_timestamps True --language en --beep_off ^
  --vad_filter true --condition_on_previous_text false >nul 2>&1
if errorlevel 1 goto :error
if not exist "%WHISPER_JSON%" ( echo [ERROR] transcribe completed but %WHISPER_JSON% missing & goto :error )

REM -- Step 1b: Resegment JSON into broadcast-paced VTT (matches Modal stable-ts) -
if exist "%BIN%\vtt_resegment.exe" (
  "%BIN%\vtt_resegment.exe" --input "%WHISPER_JSON%" --output "%VTT%" >nul 2>&1
  if errorlevel 1 ( echo [warn] resegment failed; falling back to raw Whisper VTT )
)
REM Fallback: if resegmenter missing or failed, ask Whisper to also emit VTT.
if not exist "%VTT%" (
  "%BIN%\faster-whisper-xxl.exe" "%INPUT%" ^
    --model large-v2 --output_dir "%AUDIT_DIR_NOSLASH%" --output_format vtt ^
    --word_timestamps True --language en --beep_off ^
    --vad_filter true --condition_on_previous_text false >nul 2>&1
)
if not exist "%VTT%" ( echo [ERROR] %VTT% missing after resegment + fallback & goto :error )
echo       OK

REM -- Step 2: Pre-encode QC + SCC sidecar ----------------------------------
if "%HAS_VTT2SCC%"=="1" (
  echo [2/6] Running pre-encode QC...
  "%BIN%\vtt_to_scc.exe" "%VTT%" "%SCC%" --report "%QC%" --cleaned-vtt "%VTT_CLEAN%" --cleaned-srt "%SRT%" >nul 2>&1
  if errorlevel 1 echo       QC found errors that survived auto-fix - see proof PDF
  if exist "%VTT_CLEAN%" (
    copy /y "%VTT_CLEAN%" "%VTT%" >nul
    del "%VTT_CLEAN%"
  )
  echo       OK
) else (
  echo [2/6] Skipping QC - support tool missing
)

REM -- Step 3: probe codec --------------------------------------------------
echo [3/6] Probing source codec...
for /f "delims=" %%C in ('""%BIN%\ffprobe.exe" -v error -select_streams v:0 -show_entries stream^=codec_name -of default^=nw^=1:nk^=1 "%INPUT%""') do set "VCODEC=%%C"
if not defined VCODEC set "VCODEC=unknown"
REM -bsf:v filter_units=remove_types=6 strips ALL SEI NALs from the source
REM bitstream before our injector runs. Without it, an H.264 source that
REM already carried CEA-608 in SEI would end up with both old + new captions
REM after -c:v copy. The re-encode branch doesn't need the filter — libx264
REM doesn't carry forward source SEI on its own.
if /i "%VCODEC%"=="h264" (
  set "FIRST_VIDEO=-c:v copy -bsf:v filter_units=remove_types=6"
) else (
  set "FIRST_VIDEO=-c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
)
echo       Codec: %VCODEC%

REM -- Step 4: SEI inject pipeline ------------------------------------------
echo [4/6] Injecting CEA-608 captions...
"%BIN%\ffmpeg.exe" -y -hide_banner -loglevel error -i "%INPUT%" %FIRST_VIDEO% -c:a aac -ac 2 -ar 44100 -f flv pipe:1 2>nul | "%BIN%\hybridCC-vod.exe" "%VTT%" 2>nul | "%BIN%\ffmpeg.exe" -y -hide_banner -loglevel error -f flv -i pipe:0 -c:v copy -c:a copy -a53cc 1 -movflags +faststart "%OUT%" 2>nul
if errorlevel 1 goto :error
if not exist "%OUT%" ( echo [ERROR] Inject completed but %OUT% missing & goto :error )

REM ---- Replace the source with the captioned version (no .cc suffix) -----
set "FINAL=%DIR%%STEM%.mp4"
del "%INPUT%" 2>nul
if exist "%INPUT%" (
  echo [WARN] Could not delete original "%INPUT%" - captioned copy left at %OUT%.
  set "FINAL=%OUT%"
) else (
  move /y "%OUT%" "%FINAL%" >nul
  if errorlevel 1 ( echo [WARN] Could not rename %OUT% to %FINAL% & set "FINAL=%OUT%" )
)

REM -- Step 5: Render rich proof PDF (proof_render.exe + Edge headless) -----
set "HAS_PROOF=0"
if exist "%BIN%\proof_render.exe" set "HAS_PROOF=1"

if "%HAS_VTT2SCC%"=="1" if "%HAS_PROOF%"=="1" if "%HAS_EDGE%"=="1" (
  echo [5/6] Rendering proof report and verifying captions...
  REM Probe duration so the proof PDF can identify short-form content
  REM (under 60s) and surface the "fast speaker / no breaths" explanation
  REM in the WARN/FAIL callout.
  set "SRC_DUR=0"
  for /f "delims=" %%D in ('""%BIN%\ffprobe.exe" -v error -show_entries format^=duration -of default^=nw^=1:nk^=1 "%FINAL%""') do set "SRC_DUR=%%D"
  "%BIN%\proof_render.exe" ^
    --qc "%QC%" ^
    --src-name "%SRC_NAME%" ^
    --src-stem "%STEM%" ^
    --scc "%SCC%" ^
    --output "%PROOF_HTML%" ^
    --bin "%BIN%" ^
    --mp4 "%FINAL%" ^
    --source-vtt "%VTT%" ^
    --source-duration-sec "!SRC_DUR!" >nul 2>&1
  if exist "%PROOF_HTML%" (
    "%MSEDGE%" --headless --disable-gpu --print-to-pdf="%PROOF_PDF%" --print-to-pdf-no-header "file:///%PROOF_HTML:\=/%" >nul 2>&1
    echo       OK
  ) else (
    echo       [warn] proof report not generated
  )
) else (
  echo [5/6] Skipping proof report - support tools missing
)

REM -- Step 6: Bundle audit zip + relocate VTT + clean up intermediates ---
echo [6/6] Bundling audit zip and cleaning up...
if exist "%AUDIT_ZIP%" del "%AUDIT_ZIP%"
REM Build the zip file list inside PowerShell — sidesteps cmd's quote-mangling
REM with multiple positional paths. Single shell-out, single Test-Path per file.
powershell -NoProfile -Command "$files = @(); foreach ($f in @('%PROOF_PDF%','%SCC%','%SRT%')) { if (Test-Path -LiteralPath $f) { $files += $f } }; if ($files.Count -gt 0) { Compress-Archive -LiteralPath $files -DestinationPath '%AUDIT_ZIP%' -Force }"

REM Move VTT from audit/ to vtt/ folder.
if exist "%VTT%" (
  move /y "%VTT%" "%VTT_FINAL%" >nul 2>&1
  if errorlevel 1 ( echo       [warn] Could not move VTT to %VTT_DIR% & set "VTT_FINAL=%VTT%" )
)

REM Delete the loose intermediates now that they're in the zip.
REM Keep only: data/captions/audit/<stem>.audit.zip + data/captions/vtt/<stem>.vtt
if exist "%PROOF_HTML%"  del "%PROOF_HTML%"  2>nul
if exist "%PROOF_PDF%"   del "%PROOF_PDF%"   2>nul
if exist "%SCC%"         del "%SCC%"         2>nul
if exist "%SRT%"         del "%SRT%"         2>nul
if exist "%QC%"          del "%QC%"          2>nul
if exist "%WHISPER_JSON%" del "%WHISPER_JSON%" 2>nul

echo.
echo =======================================================================
echo  DONE
echo =======================================================================
echo  MP4:    %FINAL%
if exist "%VTT_FINAL%"  echo  VTT:    %VTT_FINAL%
if exist "%AUDIT_ZIP%"  echo  Audit:  %AUDIT_ZIP%
echo.
if "%HAS_VTT2SCC%"=="0" echo  NOTE: QC tool not present - SCC and PDF skipped.
echo.
pause
exit /b 0

:error
echo.
echo =======================================================================
echo  PIPELINE FAILED
echo =======================================================================
pause
exit /b 1
