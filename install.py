import launch, torch


if not launch.is_installed("cupy"):
    cuda_version = int(round(float(torch.version.cuda)*10))
    if cuda_version <= 118:
        launch.run_pip("install cupy-cuda11x", "requirements for FastBlend (cupy)")
    else:
        launch.run_pip("install cupy-cuda12x", "requirements for FastBlend (cupy)")

if not launch.is_installed("imageio"):
    launch.run_pip("install imageio", "requirements for FastBlend (imageio)")

if not launch.is_installed("imageio_ffmpeg"):
    launch.run_pip("install imageio[ffmpeg]", "requirements for FastBlend (imageio[ffmpeg])")

if not launch.is_installed("cv2") and not launch.is_installed("opencv-python-headless"):
    launch.run_pip("install opencv-python-headless", "requirements for FastBlend (opencv-python-headless)")
