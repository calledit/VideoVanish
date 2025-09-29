echo windows installer for VideoVanish

set "CONDA=%UserProfile%\miniconda3\condabin\conda.bat"
CALL "%UserProfile%\miniconda3\Scripts\activate.bat" videovanish

echo Starting videovanish... Please wait.. It may take a few minutes before the GUI becomes visible the first run as packages need to be compiled the first time
set KMP_DUPLICATE_LIB_OK=TRUE
python videovanish.py 
pause
