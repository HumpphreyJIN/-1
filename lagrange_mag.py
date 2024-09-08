import numpy as np
from scipy.interpolate import lagrange
from data import read_video, save_video
from tqdm import tqdm


def lagrange_magnification(frames, alpha):
    magnified_frames = []
    n_frames = len(frames)
    indices = np.arange(n_frames)

    for i in tqdm(range(n_frames), desc=f"Lagrange Magnification (alpha={alpha})"):
        frame = frames[i]
        magnified_frame = np.zeros_like(frame)
        for j in range(frame.shape[2]):  # 针对每个颜色通道
            for k in range(frame.shape[1]):  # 针对每个列
                f = frame[:, k, j].astype(np.float64)  # 转换为浮点数，防止溢出
                f = (f - np.mean(f)) / np.std(f)

                # 修改部分：处理插值溢出和形状不匹配问题
                try:
                    polynomial = lagrange(indices, f)
                    interpolated_values = polynomial(indices * alpha)

                    # 如果 interpolated_values 长度大于原始数据长度，进行截断
                    if len(interpolated_values) > len(f):
                        interpolated_values = interpolated_values[:len(f)]

                    # 确保插值后的值不会溢出，并处理形状不匹配问题
                    magnified_frame[:, k, j] = np.clip(interpolated_values, 0, 255)
                except ValueError as e:
                    # 如果出现错误，直接使用原始值
                    print(f"ValueError at frame {i}, column {k}, channel {j}: {e}")
                    magnified_frame[:, k, j] = f

        magnified_frames.append(magnified_frame.astype(np.uint8))

    return magnified_frames


def apply_lagrange_magnification(input_video_path, output_video_base_path, alphas):
    frames, fps, frame_size = read_video(input_video_path)

    for alpha in alphas:
        print(f"Processing with alpha = {alpha}")
        amplified_frames = lagrange_magnification(frames, alpha)
        output_video_path = f"{output_video_base_path}{alpha}x.mp4"
        save_video(amplified_frames, output_video_path, fps, frame_size)

