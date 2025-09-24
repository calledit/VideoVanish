echo windows installer for VideoVanish

set "CONDA=%UserProfile%\miniconda3\condabin\conda.bat"
CALL "%UserProfile%\miniconda3\Scripts\activate.bat" videovanish


set KMP_DUPLICATE_LIB_OK=TRUE
python videovanish.py 
pause