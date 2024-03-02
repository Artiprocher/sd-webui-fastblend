extern "C" __global__
void pairwise_patch_error(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const int pad_size,
    const float* source_a,
    const int* nnf_a,
    const float* source_b,
    const int* nnf_b,
    float* error
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockIdx.z * (height + pad_size * 2) * (width + pad_size * 2) * channel;
    if (x >= height or y >= width) return;
    const int z_nnf = blockIdx.z * height * width * 2 + (x * width + y) * 2;
    const int x_a = nnf_a[z_nnf + 0];
    const int y_a = nnf_a[z_nnf + 1];
    const int x_b = nnf_b[z_nnf + 0];
    const int y_b = nnf_b[z_nnf + 1];
    float e = 0;
    for (int px = -r; px <= r; px++){
        for (int py = -r; py <= r; py++){
            const int pid_a = (x_a + pad_size + px) * (width + pad_size * 2) + y_a + pad_size + py;
            const int pid_b = (x_b + pad_size + px) * (width + pad_size * 2) + y_b + pad_size + py;
            for (int c = 0; c < channel; c++){
                const float diff = source_a[z + pid_a * channel + c] - source_b[z + pid_b * channel + c];
                e += diff * diff;
            }
        }
    }
    error[blockIdx.z * height * width + x * width + y] = e;
}