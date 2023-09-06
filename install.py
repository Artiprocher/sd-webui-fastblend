import launch, torch


if not launch.is_installed("cupy"):
    cuda_version = int(round(float(torch.version.cuda)*10))
    if cuda_version <= 102:
        launch.run_pip("install cupy-cuda102", "requirements for FastBlend (cupy)")
    elif cuda_version <= 110:
        launch.run_pip("install cupy-cuda110", "requirements for FastBlend (cupy)")
    elif cuda_version <= 111:
        launch.run_pip("install cupy-cuda111", "requirements for FastBlend (cupy)")
    elif cuda_version <= 118:
        launch.run_pip("install cupy-cuda11x", "requirements for FastBlend (cupy)")
    else:
        launch.run_pip("install cupy-cuda12x", "requirements for FastBlend (cupy)")

if not launch.is_installed("imageio-ffmpeg"):
    launch.run_pip("install imageio imageio[ffmpeg]", "requirements for FastBlend (imageio)")

if not launch.is_installed("cv2") and not launch.is_installed("opencv-python-headless"):
    launch.run_pip("install opencv-python-headless", "requirements for FastBlend (opencv-python-headless)")
