## Why VideoVanish?
Video object removal is one of the most powerful yet frustrating AI workflows today.  
Most existing solutions are:
- **Cloud-based** → slow uploads, privacy concerns, and limits on video length.  
- **Research demos** → amazing results, but painful to install, run, or scale beyond short clips.  
- **Image-only tools** → great for photos, but useless for long videos (flicker, no temporal consistency).  

**VideoVanish** bridges this gap by providing:
- A **local-first** workflow — no uploading, no hidden costs.  
- A **simple installer and GUI** — no command line required.  
- **State-of-the-art AI models** (e.g. DiffuEraser, SAM2) under the hood.  
- Support for **long videos** with chunking, overlap and blending.
- Both **GUI** for ease of use and **CLI** for automation.  

In short: *professional-grade AI video inpainting, without the research-paper headaches.*

### Screenshot
<img alt="videovanish" src="https://github.com/user-attachments/assets/b61c700e-7eae-43a0-be1a-a62cb1de2418" />


## Plan
VideoVanish is intended to be a **user-friendly** tool for state-of-the-art video object removal and masking.

### Intended features
- A simple installer that does not require the command line.  
  - Example: a `.bat` file for Windows and a `pip install` package for Ubuntu/Debian.  
- A basic video manipulation GUI, similar to *Topaz AI*:  
  - Timeline underneath, video preview on top.  
- **Not** browser-based.  
  - Browser-based video editing is a nightmare (it can be solved, but not worth it).  
  - Will also **not** use ComfyUI.  
- GUI will show the selected color video with the mask video overlaid on top.  
- In the GUI, you can select a **color video file** and a **mask video file**.  
- If you don’t have a mask video, the GUI will let you create one using **SAM2**:  
  - Left click = positive, right click = negative.  
  - Or drag a box to define an area.  
  - Then click **Generate Mask** to create the mask video.  
- When you have both a color video and a mask video, press the **Vanish** button:  
  - **DiffuEraser** (or whichever model is best at the time; PRs welcome) will remove the object.  
- Before pressing **Vanish**, you can set the **inference resolution**.  
  - Default = **LOW**.  
  - After inpainting, the result will be rescaled to the original resolution.  
  - The inpainted region will be sliced in and merged into a copy of the original video.  
- **Dependencies**:  
  - Both the **SAM2 model** and the **DiffuEraser model** must be installed at install time.  
- **Long videos**:  
  - Must be split into chunks with overlapping frames to maintain temporal consistency.  
  - Overlaps will be blended during stitching.  
- **Command line support**:  
  - All actions should also be available from the CLI.  
  - Example:  
    ```bash
    videovanish --color_video some_file.mkv --mask_video some_mask.mkv
    ```
