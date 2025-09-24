# VideoVanish

## Why VideoVanish?
Video object removal is one of the most powerful yet frustrating AI workflows today.  

Most existing solutions are:
- **Cloud-based** → slow uploads, privacy concerns, video length limits.  
- **Research demos** → impressive results, but painful to install or scale.  
- **Image-only tools** → fine for photos, but fail on videos (flicker, no temporal consistency).  

**VideoVanish** bridges this gap by offering:
- A **local-first** workflow — no uploads, no hidden costs.  
- A **simple installer + GUI** — no command line needed.  
- **State-of-the-art AI models** (DiffuEraser, SAM2) built-in.  
- Support for **long videos** with chunking, overlap, and blending.  
- Both **GUI** for ease of use and **CLI** for automation.  

👉 In short: *professional-grade AI video inpainting, without the research-paper headaches.*

### Screenshot
<img alt="videovanish" src="https://github.com/user-attachments/assets/b61c700e-7eae-43a0-be1a-a62cb1de2418" />

---

## Install

### Windows (GPU with plenty of VRAM recommended)
1. Install [Miniconda](https://docs.conda.io/en/latest/) with latest [installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) (choose all defaults).  
2. [Download VideoVanish (main.zip)](https://github.com/calledit/VideoVanish/archive/refs/heads/main.zip) and extract it anywhere.  
3. Double-click **`windows_install.bat`**.  
4. Double-click **`start_videovanish.bat`** to launch.  

### Linux
```bash
git clone https://github.com/calledit/VideoVanish
cd VideoVanish
./install_videovanish.sh
conda activate videovanish
python videovanish.py
```
---

## Tutorial
https://youtu.be/GMFwWv1zrVM

## Project Plan

VideoVanish is intended to be a **user-friendly** tool for state-of-the-art video object removal and masking.

### Current Features
- [x] Simple installer, no command line required.  
- [x] Basic GUI for video editing (timeline + preview).  
- [x] **Not** browser-based, and does **not** use ComfyUI.  
- [x] Load **color video** + optional **mask video**.  
- [x] If no mask video, create one in GUI using **SAM2**:  
  - Left click = add point  
  - Right click = remove point/box  
  - Drag = define area  
  - Click **Generate Mask** to build mask video  
- [x] With both color + mask, press **Vanish** → uses **DiffuEraser** to remove objects.  
- [x] Adjustable **inference resolution** (default: low).  
  - Result is rescaled and blended into original video.  
- [x] Dependencies (SAM2 + DiffuEraser models) installed automatically.  
- [x] **Command-line support** for automation.  

---

## TODO / Roadmap
- Hide console window on startup (many users find it annoying).  
  - Keep a way to view weight download progress (inline console window or GUI download manager).  
- Improve **user experience**: create a YouTube video tutorial.
- Split videos into chunks with overlapping frames to reduce vram requirements.  Overlaps should be blended during stitching. ⚠️ Perfect blending may not always be possible.  

---
