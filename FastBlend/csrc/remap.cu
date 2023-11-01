extern "C"
__global__ void remap(const int height, const int width, const int channel,
                      const int patch_size, const int pad_size,
                      const float *source_style, const int *nnf, float *target_style) {
  const int r = (patch_size - 1) / 2;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= height or y >= width)
    return;
  const int z =
      blockIdx.z * (height + pad_size * 2) * (width + pad_size * 2) * channel;
  const int pid = (x + pad_size) * (width + pad_size * 2) + (y + pad_size);
  const int min_px = x < r ? -x : -r;
  const int max_px = x + r > height - 1 ? height - 1 - x : r;
  const int min_py = y < r ? -y : -r;
  const int max_py = y + r > width - 1 ? width - 1 - y : r;
  int num = 0;
  int imsize = height * width * 2;
  for (int px = min_px; px <= max_px; px++) {
    for (int py = min_py; py <= max_py; py++) {
      const int nid = (x + px) * width + y + py;
      const int x_ = nnf[blockIdx.z * imsize + nid * 2 + 0] - px;
      const int y_ = nnf[blockIdx.z * imsize + nid * 2 + 1] - py;
      if (x_ < 0 or y_ < 0 or x_ >= height or y_ >= width)
        continue;
      const int pid_ =
          (x_ + pad_size) * (width + pad_size * 2) + (y_ + pad_size);
      num++;
      for (int c = 0; c < channel; c++) {
        target_style[z + pid * channel + c] +=
            source_style[z + pid_ * channel + c];
      }
    }
  }
  for (int c = 0; c < channel; c++) {
    target_style[z + pid * channel + c] /= num;
  }
}
