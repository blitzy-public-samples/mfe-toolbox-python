@ECHO OFF

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

REM Check for sphinx-autobuild for livehtml target
if "%1" == "livehtml" (
    where sphinx-autobuild >nul 2>&1
    if errorlevel 1 (
        echo Error: sphinx-autobuild not found. Install with pip install sphinx-autobuild.
        exit /b 1
    )
    sphinx-autobuild "%SOURCEDIR%" "%BUILDDIR%\html" %SPHINXOPTS% %O% --open-browser
    goto end
)

REM Clean target to remove build artifacts
if "%1" == "clean" (
    if exist "%BUILDDIR%\*" rmdir /s /q "%BUILDDIR%"
    if exist "api\generated\*" rmdir /s /q "api\generated"
    echo Build directory cleaned.
    goto end
)

REM Check for broken links
if "%1" == "linkcheck" (
    %SPHINXBUILD% -b linkcheck "%SOURCEDIR%" "%BUILDDIR%\linkcheck" %SPHINXOPTS% %O%
    echo Link check complete; look for any errors in the above output or in %BUILDDIR%\linkcheck\output.txt.
    goto end
)

REM API documentation generation
if "%1" == "apidoc" (
    sphinx-apidoc -o api\generated ..\mfe -f -e -M
    echo API documentation generated.
    goto end
)

REM Full build: clean, apidoc, and html
if "%1" == "fullbuild" (
    call make.bat clean
    call make.bat apidoc
    call make.bat html
    echo Full documentation build complete.
    goto end
)

REM PDF output via LaTeX
if "%1" == "pdf" (
    %SPHINXBUILD% -M latexpdf "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
    echo PDF build finished. The PDF file is in %BUILDDIR%\latex.
    goto end
)

REM EPUB output
if "%1" == "epub" (
    %SPHINXBUILD% -M epub "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
    echo EPUB build finished. The EPUB file is in %BUILDDIR%\epub.
    goto end
)

REM Single HTML page output
if "%1" == "singlehtml" (
    %SPHINXBUILD% -M singlehtml "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
    echo Single HTML build finished. The HTML file is in %BUILDDIR%\singlehtml.
    goto end
)

REM Handle standard sphinx commands
%SPHINXBUILD% -M %1 "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%

:end