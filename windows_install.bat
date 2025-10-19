echo windows installer for VideoVanish

set "CONDA=%UserProfile%\miniconda3\condabin\conda.bat"

CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

CALL "%CONDA%" create -n videovanish python=3.11 -y
CALL "%UserProfile%\miniconda3\Scripts\activate.bat" videovanish



CALL "%CONDA%" install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

REM --- Download and unpack sam2_numpy_frames ---
curl -L -o sam2.zip https://github.com/calledit/sam2_numpy_frames/archive/refs/heads/main.zip
powershell -command "Expand-Archive -Path 'sam2.zip' -DestinationPath '.' -Force"
rename sam2_numpy_frames-main sam2_numpy_frames
pushd sam2_numpy_frames
REM install requirments for sam2
pip install -e .
popd

REM Download SAM2 checkpoint
curl -L -o sam2_numpy_frames\checkpoints\sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

REM --- Download and unpack DiffuEraser_np_array  ---
curl -L -o de.zip https://github.com/calledit/DiffuEraser_np_array/archive/refs/heads/master.zip
powershell -command "Expand-Archive -Path 'de.zip' -DestinationPath '.' -Force"
rename DiffuEraser_np_array-master DiffuEraser_np_array

REM install requirments for DiffuEraser
pip install einops>=0.8.1 diffusers==0.29.2 transformers>=4.57.1 scipy matplotlib accelerate>=1.10.1 peft>=0.17.1

pip install numpy opencv-python PySide6

echo Installation done
pause
