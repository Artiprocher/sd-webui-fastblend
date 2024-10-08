# FastBlend: a Model-Free Algorithm That Can Make Your Video Smooth

This is a model-free algorithm that can make your video smooth. You can remove the flicker in your video, or render a fluent video using a series of keyframes.

[paper](https://arxiv.org/abs/2311.09265)

[ConfyUI-extension](https://github.com/AInseven/ComfyUI-fastblend)

## Usage

### Use FastBlend in [Stable-Diffusion-Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/Artiprocher/sd-webui-fastblend.git` to "URL for extension's git repository".
4. Press "Install" button.
5. Wait for 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\sd-webui-fastblend. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI".
7. You can see a `FastBlend` tab in the webui.
8. Enjoy the coherent and fluent videos!

### Use FastBlend in the Independent Webui

Install the packages in your Python environment:

```
pip install gradio numpy imageio imageio[ffmpeg] opencv-python-headless tqdm cupy-cuda12x
```

If your CUDA version is not v11.2 ~ 11.8, please read [this document](https://docs.cupy.dev/en/stable/install.html) and install the corressponding version of cupy.

Then run the following command to launch the independent webui:

```
python independent_webui.py
```

### Use FastBlend in your Python code

Install the packages as we described above.

```python3
from FastBlend.api import smooth_video, interpolate_video

# Blend
smooth_video(
    video_guide = "guide_video.mp4",
    video_guide_folder = None,
    video_style = "style_video.mp4",
    video_style_folder = None,
    mode = "Fast",
    window_size = 15,
    batch_size = 16,
    tracking_window_size = 0,
    output_path = "output_folder",
    fps = None,
    minimum_patch_size = 5,
    num_iter = 5,
    guide_weight = 10.0,
    initialize = "identity"
)

# Interpolate
interpolate_video(
    frames_path = "frames_folder",
    keyframes_path = "keyframes_folder",
    output_path = "output_folder",
    fps = None,
    batch_size = 16,
    tracking_window_size = 1,
    minimum_patch_size = 15,
    num_iter = 5,
    guide_weight = 10.0,
    initialize = "identity"
)
```

## Example

### Blend

https://github.com/Artiprocher/sd-webui-fastblend/assets/35051019/208d902d-6aba-48d7-b7d5-cd120ebd306d

1. The original video is [here](https://www.bilibili.com/video/BV1K14y1Z7cp/). We only use the first 236 frames.
2. Re-render each frame independently. The parameters are
   1. Prompt: masterpiece, best quality, anime screencap, cute, petite, long hair, black hair, blue eyes, hoodie, breasts, smile, short sleeves, hands, blue bowknot, wind, depth of field, forest, close-up,
   2. Negative prompt: (worst quality, low quality:1.4), monochrome, zombie, (interlocked fingers:1.2), extra arms,
   3. Steps: 20,
   4. Sampler: DPM++ 2M Karras,
   5. CFG scale: 7,
   6. Seed: 3010302656,
   7. Size: 768x512,
   8. Model hash: 4c79dd451a,
   9. Model: aingdiffusion_v90,
   10. Denoising strength: 1,
   11. Clip skip: 2,
   12. ControlNet 0: "Module: tile_resample, Model: control_v11f1e_sd15_tile [a371b31b], Weight: 0.4, Resize Mode: Crop and Resize, Low Vram: False, Threshold A: 1, Guidance Start: 0, Guidance End: 1, Pixel Perfect: True, Control Mode: Balanced",
   13. ControlNet 1: "Module: softedge_pidinet, Model: control_v11p_sd15_softedge [a8575a2a], Weight: 1, Resize Mode: Crop and Resize, Low Vram: False, Processor Res: 512, Guidance Start: 0, Guidance End: 1, Pixel Perfect: True, Control Mode: Balanced",
   14. ControlNet 2: "Module: depth_midas, Model: control_v11f1p_sd15_depth [cfd03158], Weight: 1, Resize Mode: Crop and Resize, Low Vram: False, Processor Res: 512, Guidance Start: 0, Guidance End: 1, Pixel Perfect: True, Control Mode: Balanced",
   15. Version: v1.6.0
3. Open "FastBlend" tab. Upload the original video to "Guide video". Upload the re-rendered video to "Style video". We use the following settings:
   1. Inference mode: Fast
   2. Sliding window size: 30
   3. Batch size: 8
   4. Minimum patch size (odd number): 5
   5. Number of iterations: 5
   6. Guide weight: 10.0
4. Click "Run". Wait a minute... (I tested this extension on an Nvidia RTX3060 laptop. It cost 12 minutes.)
5. Now you have obtained a fluent video. Go to "Extras" to upscale it using "R-ESRGAN 4+ Anime6B".

### Interpolate

https://github.com/Artiprocher/sd-webui-fastblend/assets/35051019/3490c5b4-8f67-478f-86de-f9adc2ace16a

1. The original video is [here](https://www.bilibili.com/video/BV19P411p7Gf/). We only use the frames 1108-1458. Please resize the frames to 512*512.
2. Re-render the keyframes (1108, 1140, 1172, 1204, 1236, 1268, 1300, 1332, 1364, 1396, 1428, 1458) independently. The parameters are
   1. Prompt: masterpiece, best quality, a woman, anime, flat, red hair, short hair, simple black background, bare shoulder
   2. Negative prompt: easynegative
   3. Steps: 20,
   4. Sampler: DPM++ 2M Karras,
   5. CFG scale: 7,
   6. Seed: 1,
   7. Size: 768x768,
   8. Model hash: 4c79dd451a,
   9. Model: aingdiffusion_v90,
   10. Denoising strength: 0.7,
   11. ControlNet 0: "Module: softedge_pidinet, Model: control_v11p_sd15_softedge [a8575a2a], Weight: 1, Resize Mode: Crop and Resize, Low Vram: False, Processor Res: 512, Guidance Start: 0, Guidance End: 1, Pixel Perfect: True, Control Mode: Balanced",
   12. TI hashes: "easynegative: c74b4e810b03",
   13. Version: v1.6.0
3. Open "FastBlend" tab. Click "Interpolate". Fill in the directory of original frames and the rendered keyframes. We use the following settings:
   1. Batch size: 8
   2. Tracking window size: 0
   3. Minimum patch size (odd number, larger is better): 15
   4. Number of iterations: 5
   5. Guide weight: 10
   6. NNF initialization: identity
4. Click "Run". Wait a minute... (I tested this extension on an Nvidia RTX3060 laptop. It cost 3 minutes.)
5. Now you have obtained a fluent video. Go to "Extras" to upscale it using "R-ESRGAN 4+ Anime6B".

### Compare FastBlend with CoDeF

https://github.com/Artiprocher/sd-webui-fastblend/assets/35051019/f628426d-caee-493a-b7e7-0372150f3ce1

We found an interesting project called [CoDeF](https://github.com/qiuyu96/CoDeF), which uses only one keyframe to render a video. We collected some videos from [their project page](https://qiuyu96.github.io/CoDeF/) and compared FastBlend with CoDeF. For each video, we select one keyframe from the rendered video and use this keyframe to rerender the video in interpolation mode. The parameters are

1. Batch size: 48
2. Tracking Window Size: 1
3. Minimum patch size (odd number, larger is better): 25
4. Number of iterations: 5
5. Guide weight: 10
6. NNF initialization: identity

As we can see, FastBlend is competitive with CoDeF. What's more, FastBlend is very efficiency. We only need one minute to render a video clip!

## Reference

### Blend

* Output directory: the directory to save the video.
* Inference mode

|Mode|Time|Memory|Quality|Frame by frame output|Description|
|-|-|-|-|-|-|
|Fast|■|■■■|■■|No|Blend the frames using a tree-like data structure, which requires much RAM but is fast.|
|Balanced|■■|■|■■|Yes|Blend the frames naively.|
|Accurate|■■■|■|■■■|Yes|Blend the frames and align them together for higher video quality. When [batch size] >= [sliding window size] * 2 + 1, the performance is the best.|

* Sliding window size: our algorithm will blend the frames in a sliding windows. If the size is n, each frame will be blended with the last n frames and the next n frames. A large sliding window can make the video fluent but sometimes smoggy.
* Batch size: a larger batch size makes the program faster but requires more VRAM.
* Tracking window size (only for accurate mode): The size of window in which our algorithm tracks moving objects. Empirically, 1 is enough.
* Advanced settings
    * Minimum patch size (odd number): the minimum patch size used for patch matching. (Default: 5)
    * Number of iterations: the number of iterations of patch matching. (Default: 5)
    * Guide weight: a parameter that determines how much motion feature applied to the style video. (Default: 10)
    * NNF initialization: how to initialize the NNF (Nearest Neighbor Field). (Default: identity)

### Interpolate

* Output directory: the directory to save the video.
* Batch size: a larger batch size makes the program faster but requires more VRAM.
* Tracking window size (only for accurate mode): The size of window in which our algorithm tracks moving objects. Empirically, 1 is enough.
* Advanced settings
    * Minimum patch size (odd number): the minimum patch size used for patch matching. **This parameter should be larger than that in blending. (Default: 15)**
    * Number of iterations: the number of iterations of patch matching. (Default: 5)
    * Guide weight: a parameter that determines how much motion feature applied to the style video. (Default: 10)
    * NNF initialization: how to initialize the NNF (Nearest Neighbor Field). (Default: identity)

### Cite this project

```
@article{duan2023fastblend,
  title={FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier},
  author={Duan, Zhongjie and Wang, Chengyu and Chen, Cen and Qian, Weining and Huang, Jun and Jin, Mingyi},
  journal={arXiv preprint arXiv:2311.09265},
  year={2023}
}
```
