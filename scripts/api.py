import gradio as gr
from modules import script_callbacks
import cv2, functools, imageio, os
import numpy as np
import cupy as cp
from PIL import Image
from tqdm import tqdm


remapping_kernel = cp.RawKernel(r'''
extern "C" __global__
void remap(
    const int height,
    const int width,
    const int channel,
    const int patch_size,
    const int pad_size,
    const float* source_style,
    const int* nnf,
    float* target_style
) {
    const int r = (patch_size - 1) / 2;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= height or y >= width) return;
    const int z = blockIdx.z * (height + pad_size * 2) * (width + pad_size * 2) * channel;
    const int pid = (x + pad_size) * (width + pad_size * 2) + (y + pad_size);
    const int min_px = x < r ? -x : -r;
    const int max_px = x + r > height - 1 ? height - 1 - x : r;
    const int min_py = y < r ? -y : -r;
    const int max_py = y + r > width - 1 ? width - 1 - y : r;
    for (int px = min_px; px <= max_px; px++){
        for (int py = min_py; py <= max_py; py++){
            const int nid = (x + px) * width + y + py;
            const int x_ = nnf[blockIdx.z * height * width * 2 + nid*2 + 0] - px;
            const int y_ = nnf[blockIdx.z * height * width * 2 + nid*2 + 1] - py;
            const int pid_ = (x_ + pad_size) * (width + pad_size * 2) + (y_ + pad_size);
            for (int c = 0; c < channel; c++){
                target_style[z + pid * channel + c] += source_style[z + pid_ * channel + c];
            }
        }
    }
    for (int c = 0; c < channel; c++){
        target_style[z + pid * channel + c] /= (max_px - min_px + 1) * (max_py - min_py + 1);
    }
}
''', 'remap')


patch_error_kernel = cp.RawKernel(r'''
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
''', 'patch_error')


class PatchMatcher:
    def __init__(
        self, height, width, channel, minimum_patch_size,
        threads_per_block=8, num_iter=5, gpu_id=0, guide_weight=10.0,
        random_search_steps=3, random_search_range=4,
        use_mean_target_style=False
    ):
        self.height = height
        self.width = width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.threads_per_block = threads_per_block
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.random_search_steps = random_search_steps
        self.random_search_range = random_search_range
        self.use_mean_target_style = use_mean_target_style

        self.patch_size_list = [minimum_patch_size + i*2 for i in range(num_iter)][::-1]
        self.pad_size = self.patch_size_list[0] // 2
        self.grid = (
            (height + threads_per_block - 1) // threads_per_block,
            (width + threads_per_block - 1) // threads_per_block
        )
        self.block = (threads_per_block, threads_per_block)

    def pad_image(self, image):
        return cp.pad(image, ((0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)))

    def unpad_image(self, image):
        return image[:, self.pad_size: -self.pad_size, self.pad_size: -self.pad_size, :]

    def apply_nnf_to_image(self, nnf, source):
        batch_size = source.shape[0]
        target = cp.zeros((batch_size, self.height + self.pad_size * 2, self.width + self.pad_size * 2, self.channel), dtype=cp.float32)
        remapping_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source, nnf, target)
        )
        return target

    def get_patch_error(self, source, nnf, target):
        batch_size = source.shape[0]
        error = cp.zeros((batch_size, self.height, self.width), dtype=cp.float32)
        patch_error_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source, nnf, target, error)
        )
        return error

    def get_error(self, source_guide, target_guide, source_style, target_style, nnf):
        if self.use_mean_target_style:
            target_style = self.apply_nnf_to_image(nnf, source_style)
            target_style = target_style.mean(axis=0, keepdims=True)
            target_style = target_style.repeat(source_guide.shape[0], axis=0)
        error_guide = self.get_patch_error(source_guide, nnf, target_guide)
        error_style = self.get_patch_error(source_style, nnf, target_style)
        error = error_guide * self.guide_weight + error_style
        return error

    def clamp_bound(self, nnf):
        nnf[:,:,:,0] = cp.clip(nnf[:,:,:,0], 0, self.height-1)
        nnf[:,:,:,1] = cp.clip(nnf[:,:,:,1], 0, self.width-1)
        return nnf

    def random_step(self, nnf, r):
        batch_size = nnf.shape[0]
        step = cp.random.randint(-r, r+1, size=(batch_size, self.height, self.width, 2), dtype=cp.int32)
        upd_nnf = self.clamp_bound(nnf + step)
        return upd_nnf

    def neighboor_step(self, nnf, d):
        if d==0:
            upd_nnf = cp.concatenate([nnf[:, :1, :], nnf[:, :-1, :]], axis=1)
            upd_nnf[:, :, :, 0] += 1
        elif d==1:
            upd_nnf = cp.concatenate([nnf[:, :, :1], nnf[:, :, :-1]], axis=2)
            upd_nnf[:, :, :, 1] += 1
        elif d==2:
            upd_nnf = cp.concatenate([nnf[:, 1:, :], nnf[:, -1:, :]], axis=1)
            upd_nnf[:, :, :, 0] -= 1
        elif d==3:
            upd_nnf = cp.concatenate([nnf[:, :, 1:], nnf[:, :, -1:]], axis=2)
            upd_nnf[:, :, :, 1] -= 1
        upd_nnf = self.clamp_bound(upd_nnf)
        return upd_nnf

    def update(self, source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf):
        upd_err = self.get_error(source_guide, target_guide, source_style, target_style, upd_nnf)
        upd_idx = (upd_err < err)
        nnf[upd_idx] = upd_nnf[upd_idx]
        err[upd_idx] = upd_err[upd_idx]
        return nnf, err

    def propagation(self, source_guide, target_guide, source_style, target_style, nnf, err):
        for d in cp.random.permutation(4):
            upd_nnf = self.neighboor_step(nnf, d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err
        
    def random_search(self, source_guide, target_guide, source_style, target_style, nnf, err):
        for i in range(self.random_search_steps):
            upd_nnf = self.random_step(nnf, self.random_search_range)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err

    def iteration(self, source_guide, target_guide, source_style, target_style, nnf, err):
        nnf, err = self.propagation(source_guide, target_guide, source_style, target_style, nnf, err)
        nnf, err = self.random_search(source_guide, target_guide, source_style, target_style, nnf, err)
        return nnf, err

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf):
        with cp.cuda.Device(self.gpu_id):
            source_guide = self.pad_image(source_guide)
            target_guide = self.pad_image(target_guide)
            source_style = self.pad_image(source_style)
            for it in range(self.num_iter):
                self.patch_size = self.patch_size_list[it]
                target_style = self.apply_nnf_to_image(nnf, source_style)
                err = self.get_error(source_guide, target_guide, source_style, target_style, nnf)
                nnf, err = self.iteration(source_guide, target_guide, source_style, target_style, nnf, err)
            target_style = self.unpad_image(self.apply_nnf_to_image(nnf, source_style))
        return nnf, target_style


class PyramidPatchMatcher:
    def __init__(self, image_height, image_width, channel, minimum_patch_size, threads_per_block=8, num_iter=5, gpu_id=0, guide_weight=10.0, use_mean_target_style=False, initialize="identity"):
        maximum_patch_size = minimum_patch_size + (num_iter - 1) * 2
        self.pyramid_level = int(np.log2(min(image_height, image_width) / maximum_patch_size))
        self.pyramid_heights = []
        self.pyramid_widths = []
        self.patch_matchers = []
        self.minimum_patch_size = minimum_patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.initialize = initialize
        for level in range(self.pyramid_level):
            height = image_height//(2**(self.pyramid_level - 1 - level))
            width = image_width//(2**(self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(PatchMatcher(
                height, width, channel, minimum_patch_size=minimum_patch_size,
                threads_per_block=threads_per_block, num_iter=num_iter, gpu_id=gpu_id, guide_weight=guide_weight,
                use_mean_target_style=use_mean_target_style
            ))

    def resample_image(self, images, level):
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        images = images.get()
        images_resample = []
        for image in images:
            image_resample = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            images_resample.append(image_resample)
        images_resample = cp.array(np.stack(images_resample), dtype=cp.float32)
        return images_resample

    def initialize_nnf(self, batch_size):
        if self.initialize == "random":
            height, width = self.pyramid_heights[0], self.pyramid_widths[0]
            nnf = cp.stack([
                cp.random.randint(0, height, (batch_size, height, width), dtype=cp.int32),
                cp.random.randint(0, width, (batch_size, height, width), dtype=cp.int32)
            ], axis=3)
        elif self.initialize == "identity":
            height, width = self.pyramid_heights[0], self.pyramid_widths[0]
            nnf = cp.stack([
                cp.repeat(cp.arange(height), width).reshape(height, width),
                cp.tile(cp.arange(width), height).reshape(height, width)
            ], axis=2)
            nnf = cp.stack([nnf] * batch_size)
        else:
            raise NotImplementedError()
        return nnf

    def update_nnf(self, nnf, level):
        # upscale
        nnf = nnf.repeat(2, axis=1).repeat(2, axis=2) * 2
        nnf[:,[i for i in range(nnf.shape[0]) if i&1],:,0] += 1
        nnf[:,:,[i for i in range(nnf.shape[0]) if i&1],1] += 1
        # check if scale is 2
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        if height != nnf.shape[0] * 2 or width != nnf.shape[1] * 2:
            nnf = nnf.get().astype(np.float32)
            nnf = [cv2.resize(n, (width, height), interpolation=cv2.INTER_LINEAR) for n in nnf]
            nnf = cp.array(np.stack(nnf), dtype=cp.int32)
            nnf = self.patch_matchers[level].clamp_bound(nnf)
        return nnf

    def apply_nnf_to_image(self, nnf, image):
        with cp.cuda.Device(self.gpu_id):
            image = self.patch_matchers[-1].pad_image(image)
            image = self.patch_matchers[-1].apply_nnf_to_image(nnf, image)
        return image

    def estimate_nnf(self, source_guide, target_guide, source_style):
        with cp.cuda.Device(self.gpu_id):
            if not isinstance(source_guide, cp.ndarray):
                source_guide = cp.array(source_guide, dtype=cp.float32)
            if not isinstance(target_guide, cp.ndarray):
                target_guide = cp.array(target_guide, dtype=cp.float32)
            if not isinstance(source_style, cp.ndarray):
                source_style = cp.array(source_style, dtype=cp.float32)
            for level in range(self.pyramid_level):
                nnf = self.initialize_nnf(source_guide.shape[0]) if level==0 else self.update_nnf(nnf, level)
                source_guide_ = self.resample_image(source_guide, level)
                target_guide_ = self.resample_image(target_guide, level)
                source_style_ = self.resample_image(source_style, level)
                nnf, target_style = self.patch_matchers[level].estimate_nnf(
                    source_guide_, target_guide_, source_style_, nnf
                )
        return nnf.get(), target_style.get()


class TableManager:
    def __init__(self):
        pass

    def task_list(self, n):
        tasks = []
        max_level = 1
        while (1<<max_level)<=n:
            max_level += 1
        for i in range(n):
            j = i
            for level in range(max_level):
                if i&(1<<level):
                    continue
                j |= 1<<level
                if j>=n:
                    break
                meta_data = {
                    "source": i,
                    "target": j,
                    "level": level + 1
                }
                tasks.append(meta_data)
        tasks.sort(key=functools.cmp_to_key(lambda u, v: u["level"]-v["level"]))
        return tasks
    
    def build_remapping_table(self, frames_guide, frames_style, patch_match_engine, batch_size, desc=""):
        n = len(frames_guide)
        tasks = self.task_list(n)
        remapping_table = [[(frames_style[i], 1)] for i in range(n)]
        for batch_id in tqdm(range(0, len(tasks), batch_size), desc=desc):
            tasks_batch = tasks[batch_id: min(batch_id+batch_size, len(tasks))]
            source_guide = np.stack([frames_guide[task["source"]] for task in tasks_batch])
            target_guide = np.stack([frames_guide[task["target"]] for task in tasks_batch])
            source_style = np.stack([frames_style[task["source"]] for task in tasks_batch])
            _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
            for task, result in zip(tasks_batch, target_style):
                target, level = task["target"], task["level"]
                if len(remapping_table[target])==level:
                    remapping_table[target].append((result, 1))
                else:
                    frame, weight = remapping_table[target][level]
                    remapping_table[target][level] = (
                        frame * (weight / (weight + 1)) + result / (weight + 1),
                        weight + 1
                    )
        return remapping_table

    def remapping_table_to_blending_table(self, table):
        for i in range(len(table)):
            for j in range(1, len(table[i])):
                frame_1, weight_1 = table[i][j-1]
                frame_2, weight_2 = table[i][j]
                frame = (frame_1 + frame_2) / 2
                weight = weight_1 + weight_2
                table[i][j] = (frame, weight)
        return table

    def tree_query(self, leftbound, rightbound):
        node_list = []
        node_index = rightbound
        while node_index>=leftbound:
            node_level = 0
            while (1<<node_level)&node_index and node_index-(1<<node_level+1)+1>=leftbound:
                node_level += 1
            node_list.append((node_index, node_level))
            node_index -= 1<<node_level
        return node_list

    def process_window_sum(self, frames_guide, blending_table, patch_match_engine, window_size, batch_size, desc=""):
        n = len(blending_table)
        tasks = []
        frames_result = []
        for target in range(n):
            node_list = self.tree_query(max(target-window_size, 0), target)
            for source, level in node_list:
                if source!=target:
                    meta_data = {
                        "source": source,
                        "target": target,
                        "level": level
                    }
                    tasks.append(meta_data)
                else:
                    frames_result.append(blending_table[target][level])
        for batch_id in tqdm(range(0, len(tasks), batch_size), desc=desc):
            tasks_batch = tasks[batch_id: min(batch_id+batch_size, len(tasks))]
            source_guide = np.stack([frames_guide[task["source"]] for task in tasks_batch])
            target_guide = np.stack([frames_guide[task["target"]] for task in tasks_batch])
            source_style = np.stack([blending_table[task["source"]][task["level"]][0] for task in tasks_batch])
            _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
            for task, frame_2 in zip(tasks_batch, target_style):
                source, target, level = task["source"], task["target"], task["level"]
                frame_1, weight_1 = frames_result[target]
                weight_2 = blending_table[source][level][1]
                weight = weight_1 + weight_2
                frame = frame_1 * (weight_1 / weight) + frame_2 * (weight_2 / weight)
                frames_result[target] = (frame, weight)
        return frames_result


class FastModeRunner:
    def __init__(self):
        pass

    def run(self, frames_guide, frames_style, batch_size, window_size, ebsynth_config, save_path=None):
        table_manager = TableManager()
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            **ebsynth_config
        )
        # left part
        table_l = table_manager.build_remapping_table(frames_guide, frames_style, patch_match_engine, batch_size, desc="Fast Mode Step 1/4")
        table_l = table_manager.remapping_table_to_blending_table(table_l)
        table_l = table_manager.process_window_sum(frames_guide, table_l, patch_match_engine, window_size, batch_size, desc="Fast Mode Step 2/4")
        # right part
        table_r = table_manager.build_remapping_table(frames_guide[::-1], frames_style[::-1], patch_match_engine, batch_size, desc="Fast Mode Step 3/4")
        table_r = table_manager.remapping_table_to_blending_table(table_r)
        table_r = table_manager.process_window_sum(frames_guide[::-1], table_r, patch_match_engine, window_size, batch_size, desc="Fast Mode Step 4/4")[::-1]
        # merge
        frames = []
        for (frame_l, weight_l), frame_m, (frame_r, weight_r) in zip(table_l, frames_style, table_r):
            weight_m = -1
            weight = weight_l + weight_m + weight_r
            frame = frame_l * (weight_l / weight) + frame_m * (weight_m / weight) + frame_r * (weight_r / weight)
            frames.append(frame)
        frames = [frame.clip(0, 255).astype("uint8") for frame in frames]
        if save_path is not None:
            for target, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(save_path, "%05d.png" % target))


class BalancedModeRunner:
    # Not visible for users
    def __init__(self):
        pass

    def run(self, frames_guide, frames_style, batch_size, window_size, ebsynth_config, desc=""):
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            **ebsynth_config
        )
        # tasks
        n = len(frames_style)
        tasks = []
        for target in range(n):
            for source in range(target - window_size, target + window_size + 1):
                if source >= 0 and source < n and source != target:
                    tasks.append((source, target))
        # run
        frames = [(frames_style[i], 1) for i in range(n)]
        for batch_id in tqdm(range(0, len(tasks), batch_size), desc=desc):
            tasks_batch = tasks[batch_id: min(batch_id+batch_size, len(tasks))]
            source_guide = np.stack([frames_guide[source] for source, target in tasks_batch])
            target_guide = np.stack([frames_guide[target] for source, target in tasks_batch])
            source_style = np.stack([frames_style[source] for source, target in tasks_batch])
            _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
            for (source, target), result in zip(tasks_batch, target_style):
                frame, weight = frames[target]
                frames[target] = (
                    frame * (weight / (weight + 1)) + result / (weight + 1),
                    weight + 1
                )
        frames = [frame for frame, weight in frames]
        frames = [frame.clip(0, 255).astype("uint8") for frame in frames]
        return frames


class AccurateModeRunner:
    def __init__(self):
        pass

    def run(self, frames_guide, frames_style, batch_size, window_size, ebsynth_config, desc="Accurate Mode", save_path=None):
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            use_mean_target_style=True,
            **ebsynth_config
        )
        # run
        n = len(frames_style)
        for target in tqdm(range(n), desc=desc):
            l, r = max(target - window_size, 0), min(target + window_size + 1, n)
            remapped_frames = []
            for i in range(l, r, batch_size):
                j = min(i + batch_size, r)
                source_guide = np.stack(frames_guide[source] for source in range(i, j))
                target_guide = np.stack([frames_guide[target]] * (j - i))
                source_style = np.stack(frames_style[source] for source in range(i, j))
                _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
                remapped_frames.append(target_style)
            frame = np.concatenate(remapped_frames, axis=0).mean(axis=0)
            frame = frame.clip(0, 255).astype("uint8")
            if save_path is not None:
                Image.fromarray(frame).save(os.path.join(save_path, "%05d.png" % target))


def read_video(file_name):
    reader = imageio.get_reader(file_name)
    video = []
    for frame in reader:
        frame = np.array(frame)
        video.append(frame)
    reader.close()
    return video


def get_video_fps(file_name):
    reader = imageio.get_reader(file_name)
    fps = reader.get_meta_data()["fps"]
    reader.close()
    return fps


def save_video(frames_path, video_path, num_frames, fps):
    writer = imageio.get_writer(video_path, fps=fps, quality=9)
    for i in range(num_frames):
        frame = np.array(Image.open(os.path.join(frames_path, "%05d.png" % i)))
        writer.append_data(frame)
    writer.close()
    return video_path


class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return np.array(self.reader.get_data(item))

    def __del__(self):
        self.reader.close()


def split_file_name(file_name):
    result = []
    number = -1
    for i in file_name:
        if ord(i)>=ord("0") and ord(i)<=ord("9"):
            if number == -1:
                number = 0
            number = number*10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    result = tuple(result)
    return result


def search_for_images(folder):
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    file_list = [os.path.join(folder, i) for i in file_list]
    return file_list


def read_images(folder):
    file_list = search_for_images(folder)
    frames = [np.array(Image.open(i)) for i in file_list]
    return frames


class LowMemoryImageFolder:
    def __init__(self, folder):
        self.file_list = search_for_images(folder)
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        return np.array(Image.open(self.file_list[item]))

    def __del__(self):
        pass


def get_video_data_structure(video_file, image_folder, low_ram):
    if video_file is not None:
        if low_ram:
            return LowMemoryVideo(video_file)
        else:
            return read_video(video_file)
    else:
        if low_ram:
            return LowMemoryImageFolder(image_folder)
        else:
            return read_images(image_folder)


def smooth_video(
    video_guide,
    video_guide_folder,
    video_style,
    video_style_folder,
    mode,
    low_ram,
    window_size,
    batch_size,
    output_path,
    minimum_patch_size,
    num_iter,
    guide_weight,
    initialize,
    progress = None,
):
    # input
    if mode=="Fast":
        low_ram = False
    frames_guide = get_video_data_structure(video_guide, video_guide_folder, low_ram)
    frames_style = get_video_data_structure(video_style, video_style_folder, low_ram)
    # output
    if output_path == "":
        if video_style is None:
            output_path = os.path.join(video_style_folder, "output")
        else:
            output_path = os.path.join(os.path.split(video_style)[0], "output")
        os.makedirs(output_path, exist_ok=True)
        print("No valid output_path. Your video will be saved here:", output_path)
    elif not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print("Your video will be saved here:", output_path)
    frames_path = os.path.join(output_path, "frames")
    video_path = os.path.join(output_path, "video.mp4")
    os.makedirs(frames_path, exist_ok=True)
    # process
    ebsynth_config = {
        "minimum_patch_size": minimum_patch_size,
        "threads_per_block": 8,
        "num_iter": num_iter,
        "gpu_id": 0,
        "guide_weight": guide_weight,
        "initialize": initialize
    }
    if mode == "Fast":
        FastModeRunner().run(frames_guide, frames_style, batch_size=batch_size, window_size=window_size, ebsynth_config=ebsynth_config, save_path=frames_path)
    elif mode == "Accurate":
        AccurateModeRunner().run(frames_guide, frames_style, batch_size=batch_size, window_size=window_size, ebsynth_config=ebsynth_config, save_path=frames_path)
    # output
    fps = get_video_fps(video_style) if video_style is not None else 30
    video_path = save_video(frames_path, video_path, num_frames=len(frames_style), fps=fps)
    return output_path, video_path


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Guide video"):
                    video_guide = gr.Video(label="Guide video")
                with gr.Tab("Guide video (images format)"):
                    video_guide_folder = gr.Textbox(label="Guide video (images format)", value="")
            with gr.Column():
                with gr.Tab("Style video"):
                    video_style = gr.Video(label="Style video")
                with gr.Tab("Style video (images format)"):
                    video_style_folder = gr.Textbox(label="Style video (images format)", value="")
            with gr.Column():
                video_output = gr.Video(label="Output video", interactive=False, show_share_button=True)
                output_path = gr.Textbox(label="Output directory", value="")
        btn = gr.Button(value="Run")
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Settings")
                mode = gr.Radio(["Fast", "Accurate"], label="Inference mode", value="Fast")
                low_ram = gr.Checkbox(label="Low RAM (only for accurate mode)", value=False)
                window_size = gr.Slider(label="Sliding window size", value=15, minimum=1, maximum=1000, step=1)
                batch_size = gr.Slider(label="Batch size", value=8, minimum=1, maximum=128, step=1)
                gr.Markdown("## Advanced Settings")
                minimum_patch_size = gr.Slider(label="Minimum patch size (odd number)", value=5, minimum=5, maximum=99, step=2)
                num_iter = gr.Slider(label="Number of iterations", value=5, minimum=1, maximum=10, step=1)
                guide_weight = gr.Slider(label="Guide weight", value=10.0, minimum=0.0, maximum=100.0, step=0.1)
                initialize = gr.Radio(["identity", "random"], label="NNF initialization", value="identity")
            with gr.Column():
                reference = gr.Markdown("""
# Reference

**Important: this extension doesn't support long videos. Please use short videos (e.g., several seconds).**

* Inference mode
    * Fast
        * Blend the frames using a tree-like data structure, which requires much RAM but is fast.
        * The time consumed is not relavant to the size of sliding window, thus you can use a very large sliding window.
        * This inference mode doesn't support low RAM.
    * Accurate
        * Blend the frames and align them together for higher video quality.
        * The time consumed is in direct proportion to the size of sliding window.
        * This inference mode supports low RAW.
        * When [batch size] >= [sliding window size] * 2 + 1, the performance is the best.
* Low RAM: if you don't have enough RAM, please use it. Note that it doesn't work in fast mode.
* Sliding window size: our algorithm will blend the frames in a sliding windows. If the size is n, each frame will be blended with the last n frames and the next n frames. A large sliding window can make the video fluent but sometimes smoggy.
* Batch size: a larger batch size makes the program faster but requires more VRAM.
* Output directory: the directory to save the video.
* Advanced settings
    * Minimum patch size (odd number): the minimum patch size used for patch matching. (Default: 5)
    * Number of iterations: the number of iterations of patch matching. (Default: 5)
    * Guide weight: a parameter that determines how much motion feature applied to the style video. (Default: 10)
    * NNF initialization: how to initialize the NNF (Nearest Neighbor Field). (Default: identity)
                """)
        btn.click(
            smooth_video,
            inputs=[
                video_guide,
                video_guide_folder,
                video_style,
                video_style_folder,
                mode,
                low_ram,
                window_size,
                batch_size,
                output_path,
                minimum_patch_size,
                num_iter,
                guide_weight,
                initialize
            ],
            outputs=[output_path, video_output]
        )
        return [(ui_component, "FastBlend", "extension_FastBlend")]

script_callbacks.on_ui_tabs(on_ui_tabs)
