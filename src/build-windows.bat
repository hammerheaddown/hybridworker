@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM Build hybridCC-vod.exe for Windows using w64devkit.
REM
REM Prereqs (one-time setup, see README):
REM   1. w64devkit unpacked at C:\Users\vr_re\w64devkit
REM   2. libcaption cloned + built at C:\Users\vr_re\hybridcc-build\libcaption
REM      (cmake -G "MinGW Makefiles" -DENABLE_RE2C=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .)
REM      (make -j4)
REM
REM Output: hybridCC-vod.exe dropped into hybridplayout\bin\
REM ─────────────────────────────────────────────────────────────────────────

setlocal

set "DEVKIT=C:\Users\vr_re\w64devkit"
set "LIBCAP=C:\Users\vr_re\hybridcc-build\libcaption"
set "SRC_DIR=%~dp0"
set "BIN_DIR=%~dp0..\..\hybridplayout\bin"

set "PATH=%DEVKIT%\bin;%PATH%"

if not exist "%DEVKIT%\bin\gcc.exe" (
  echo [ERROR] gcc.exe not found at %DEVKIT%\bin
  echo Install w64devkit per build-windows-setup.md, then re-run.
  exit /b 1
)

if not exist "%LIBCAP%\libcaption.a" (
  echo [ERROR] libcaption.a not found at %LIBCAP%
  echo Run libcaption build first (see comments at top of this file^).
  exit /b 1
)

echo [build] Compiling hybridCC-vod.exe...
gcc -O2 -Wall ^
  -I "%LIBCAP%" -I "%LIBCAP%\src" ^
  -I "%LIBCAP%\examples" -I "%LIBCAP%\caption" ^
  -o "%BIN_DIR%\hybridCC-vod.exe" ^
  "%SRC_DIR%hybridCC-vod.c" ^
  "%LIBCAP%\examples\flv.c" ^
  "%LIBCAP%\libcaption.a" ^
  -lm

if errorlevel 1 (
  echo [build] FAILED
  exit /b 1
)

echo [build] OK -^> %BIN_DIR%\hybridCC-vod.exe
"%BIN_DIR%\hybridCC-vod.exe" 2>&1 | findstr /C:"hybridCC-vod"

endlocal
