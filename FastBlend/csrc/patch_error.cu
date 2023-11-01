extern "C" __global__
void patch_error(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const int pad_size,
    const float* source,
    const int* nnf,
    const float* target,
    float* error
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockIdx.z * (height + pad_size * 2) * (width + pad_size * 2) * channel;
    if (x >= height or y >= width) return;
    const int x_ = nnf[blockIdx.z * height * width * 2 + (x * width + y)*2 + 0];
    const int y_ = nnf[blockIdx.z * height * width * 2 + (x * width + y)*2 + 1];
    float e = 0;
    for (int px = -r; px <= r; px++){
        for (int py = -r; py <= r; py++){
            const int pid = (x + pad_size + px) * (width + pad_size * 2) + y + pad_size + py;
            const int pid_ = (x_ + pad_size + px) * (width + pad_size * 2) + y_ + pad_size + py;
            for (int c = 0; c < channel; c++){
                const float diff = target[z + pid * channel + c] - source[z + pid_ * channel + c];
                e += diff * diff;
            }
        }
    }
    error[blockIdx.z * height * width + x * width + y] = e;
}