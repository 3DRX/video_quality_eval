import subprocess
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2)


def calculate_vmaf(ref_path, dist_path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            dist_path,
            "-i",
            ref_path,
            "-lavfi",
            "libvmaf=model_path=vmaf_v0.6.1.pkl",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = result.stderr
    vmaf_scores = []
    for line in output.splitlines():
        if "VMAF score" in line:
            vmaf_scores.append(float(line.split()[-1]))
    return np.mean(vmaf_scores)


def main(reference_video, distorted_video):
    ref_cap = cv2.VideoCapture(reference_video)
    dist_cap = cv2.VideoCapture(distorted_video)
    length_ref = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length_dist = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert (
        length_ref == length_dist
    ), "Reference and distorted videos should have the same length"
    print(f"Reference video length: {length_ref}")

    pbar = tqdm(total=length_ref)

    psnr_values = []
    ssim_values = []

    while True:
        ret_ref, ref_frame = ref_cap.read()
        ret_dist, dist_frame = dist_cap.read()
        assert ref_frame.shape == dist_frame.shape, "Frames should have the same shape"
        if not ret_ref or not ret_dist:
            break
        psnr_value = calculate_psnr(ref_frame, dist_frame)
        ssim_value = calculate_ssim(ref_frame, dist_frame)
        print(f"PSNR: {psnr_value}, SSIM: {ssim_value}")
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        pbar.update(1)
        pass

    ref_cap.release()
    dist_cap.release()

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    avg_vmaf = calculate_vmaf(reference_video, distorted_video)

    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average VMAF: {avg_vmaf}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <reference_video> <distorted_video>")
        sys.exit(1)

    reference_video = sys.argv[1]
    distorted_video = sys.argv[2]
    main(reference_video, distorted_video)
