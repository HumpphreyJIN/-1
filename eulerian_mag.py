import cv2
import numpy as np
from data import read_video  # 假设这是你用来读取视频的自定义函数
from tqdm import tqdm


def eulerian_magnification(frames, alpha, freq_lo, freq_hi, fps, pyramid_levels):
    frame_size = (frames[0].shape[1] // 4, frames[0].shape[0] // 4)
    original_size = (frames[0].shape[1], frames[0].shape[0])

    laplacian_pyramids = []

    for i, frame in enumerate(tqdm(frames, desc="Building Gaussian Pyramids")):
        resized_frame = cv2.resize(frame, frame_size)
        pyramid = [resized_frame]
        for _ in range(pyramid_levels - 1):
            resized_frame = cv2.pyrDown(resized_frame)
            pyramid.append(resized_frame)
        laplacian_pyramids.append(pyramid)
        print(f"Frame {i} Gaussian Pyramid Built, resized_frame min: {resized_frame.min()}, max: {resized_frame.max()}")

    filtered_pyramids = []

    for i, pyramid in enumerate(tqdm(laplacian_pyramids, desc="Filtering Pyramids")):
        target_shape = pyramid[0].shape
        pyramid_array = np.array(
            [cv2.resize(level, (target_shape[1], target_shape[0])).astype(np.float32) for level in pyramid])
        fft = np.fft.fft(pyramid_array.astype(np.complex64), axis=0)
        frequencies = np.fft.fftfreq(pyramid_array.shape[0], d=1.0 / fps)

        # 调试信息：检查频率值
        print(f"Frequencies: min={frequencies.min()}, max={frequencies.max()}")

        mask = (freq_lo <= np.abs(frequencies)) & (np.abs(frequencies) <= freq_hi)

        # 调试信息：检查频率掩码
        print(f"Frame {i} Frequency Mask: min={mask.min()}, max={mask.max()}, mask sum={mask.sum()}")

        fft[~mask] = 0
        filtered_pyramid = np.fft.ifft(fft, axis=0).real
        filtered_pyramids.append(filtered_pyramid)
        print(
            f"Frame {i} Filtered, fft min: {fft.min()}, max: {fft.max()}, filtered_pyramid min: {filtered_pyramid.min()}, max: {filtered_pyramid.max()}")

    amplified_pyramids = [pyramid * alpha for pyramid in filtered_pyramids]
    amplified_frames = []

    for i, pyramid in enumerate(tqdm(amplified_pyramids, desc="Reconstructing Amplified Frames")):
        frame = pyramid[-1]
        for level in reversed(pyramid[:-1]):
            size = (level.shape[1], level.shape[0])

            frame = cv2.pyrUp(frame)
            frame = cv2.resize(frame, (size[1], size[0]))  # 确保上采样后的尺寸匹配

            if level.shape[1] != frame.shape[1] or level.shape[0] != frame.shape[0]:
                level = cv2.resize(level, (frame.shape[1], frame.shape[0]))

            frame = cv2.add(frame, level)  # 添加当前层
        frame = cv2.resize(frame, (original_size[1], original_size[0]))  # 最后调整到原始尺寸
        amplified_frames.append(frame.astype(np.uint8))
        print(f"Frame {i} Reconstructed, frame min: {frame.min()}, max: {frame.max()}")

    return amplified_frames


def save_video(frames, output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for i, frame in enumerate(tqdm(frames, desc="Saving Video")):
        out.write(frame)
        print(f"Frame {i} Saved: shape={frame.shape}, dtype={frame.dtype}, min: {frame.min()}, max: {frame.max()}")

    out.release()
    print(f"Video saved to {output_path}")


def apply_eulerian_magnification(input_video_path, output_video_base_path, alphas, freq_lo, freq_hi):
    frames, fps, frame_size = read_video(input_video_path)
    print(f"Read {len(frames)} frames from {input_video_path} with FPS {fps} and frame size {frame_size}")

    for alpha in alphas:
        print(f"Processing with alpha = {alpha}")
        amplified_frames = eulerian_magnification(frames, alpha, freq_lo=freq_lo, freq_hi=freq_hi, fps=fps, pyramid_levels=3)

        # 检查 amplified_frames 是否正确
        print(f"Number of amplified frames: {len(amplified_frames)}")
        for i, frame in enumerate(amplified_frames[:5]):  # 只打印前5帧
            print(f"Amplified Frame {i} shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")

        output_video_path = f"{output_video_base_path}{alpha}x.mp4"
        print(f"Saving video to {output_video_path}")
        save_video(amplified_frames, output_video_path, fps, frame_size)


if __name__ == "__main__":
    input_video_path = 'e_mag/video/mini.mp4'
    output_video_base_path = 'output_eulerian_'
    alphas = [5]
    freq_lo = 0.0033 # 调整低频阈值
    freq_hi = 3.0  # 调整高频阈值
    apply_eulerian_magnification(input_video_path, output_video_base_path, alphas, freq_lo, freq_hi)







