@echo off
rem Builds nr_runtime.dll (hipcc, ROCm 6.4), nvngx.dll_dlssnr.dll and test_sync.exe (VS Build Tools 2022) into %OUT% (default: out\ next to this file).
setlocal
if "%OUT%"=="" set OUT=%~dp0out
if not exist "%OUT%" mkdir "%OUT%"
cd /d "%~dp0"
"C:\Program Files\AMD\ROCm\6.4\bin\hipcc.exe" --offload-host-only -std=c++17 -O2 -shared runtime\nr_runtime.cpp -lcabinet -o "%OUT%\nr_runtime.dll" || exit /b 1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /LD /O2 /EHsc /std:c++17 dll\dlssnr_nr.cpp /Fo"%OUT%\\" /Fe"%OUT%\nvngx.dll_dlssnr.dll" /link d3d12.lib || exit /b 1
cl /nologo /O2 /EHsc /std:c++17 dll\test_sync.cpp /Fo"%OUT%\\" /Fe"%OUT%\test_sync.exe" /link d3d12.lib dxgi.lib || exit /b 1
