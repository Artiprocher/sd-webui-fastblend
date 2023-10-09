from ..patch_match import PyramidPatchMatcher
import numpy as np
from tqdm import tqdm



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
