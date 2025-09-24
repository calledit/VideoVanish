echo windows installer for VideoVanish

set "CONDA=%UserProfile%\miniconda3\condabin\conda.bat"

CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
CALL "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

CALL "%CONDA%" create -n videovanish python=3.11 -y
CALL "%UserProfile%\miniconda3\Scripts\activate.bat" videovanish



CALL "%CONDA%" install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

git clone https://github.com/calledit/sam2_numpy_frames
cd sam2_numpy_frames
pip install -e .
cd checkpoints
curl -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..
cd..

git clone https://github.com/calledit/DiffuEraser_np_array
pip install einops diffusers==0.29.2 transformers scipy matplotlib accelerate peft

pip install numpy opencv-python PySide6

rem do i need this? KMP_DUPLICATE_LIB_OK=TRUE
