import subprocess
import sys

import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calculate_ssim(img1, img2):
    # Convert images to grayscale for SSIM calculation
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value = cv2.quality.QualitySSIM_compute(img1_gray, img2_gray)[0]  # pyright: ignore
    return ssim_value


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


def process_frame_pair(ref_frame, dist_frame):
    psnr_value = calculate_psnr(ref_frame, dist_frame)
    ssim_value = calculate_ssim(ref_frame, dist_frame)
    return psnr_value, ssim_value


def process_batch(executor, batch_ref_frames, batch_dist_frames):
    results = list(
        executor.map(process_frame_pair, batch_ref_frames, batch_dist_frames)
    )

    psnr_values, ssim_values = zip(*results)
    return psnr_values, ssim_values


def batch_process_video(ref_video_path, dist_video_path, batch_size=5000):
    ref_cap = cv2.VideoCapture(ref_video_path)
    dist_cap = cv2.VideoCapture(dist_video_path)
    ref_length = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dist_length = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert ref_length == dist_length, "Videos must have the same number of frames"
    pbar = tqdm(total=ref_length // batch_size)

    psnr_values = []
    ssim_values = []

    with ThreadPoolExecutor() as executor:
        while True:
            batch_ref_frames = []
            batch_dist_frames = []

            for _ in range(batch_size):
                ret_ref, ref_frame = ref_cap.read()
                ret_dist, dist_frame = dist_cap.read()

                if not ret_ref or not ret_dist:
                    break

                batch_ref_frames.append(ref_frame)
                batch_dist_frames.append(dist_frame)

            if not batch_ref_frames:
                break

            batch_psnr, batch_ssim = process_batch(
                executor, batch_ref_frames, batch_dist_frames
            )
            pbar.update(1)

            psnr_values.extend(batch_psnr)
            ssim_values.extend(batch_ssim)
        pass

    ref_cap.release()
    dist_cap.release()

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim


def main(reference_video, distorted_video):
    avg_psnr, avg_ssim = batch_process_video(reference_video, distorted_video)

    # avg_vmaf = calculate_vmaf(reference_video, distorted_video)
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    # print(f"Average VMAF: {avg_vmaf}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <reference_video> <distorted_video>")
        sys.exit(1)

    reference_video = sys.argv[1]
    distorted_video = sys.argv[2]
    main(reference_video, distorted_video)
