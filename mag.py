from eulerian_mag import apply_eulerian_magnification
from lagrange_mag import apply_lagrange_magnification

input_video_path = 'e_mag/video/mini.mp4'
output_video_base_path_eulerian = 'output_eulerian_'
output_video_base_path_lagrange = 'output_lagrange_'
alphas = [5]


def mag(str):

    if str == 'e':
        apply_eulerian_magnification(input_video_path, output_video_base_path_eulerian, alphas)
    elif str == 'l':
        apply_lagrange_magnification(input_video_path, output_video_base_path_lagrange, alphas)


if __name__ == '__main__':
    mag('l')