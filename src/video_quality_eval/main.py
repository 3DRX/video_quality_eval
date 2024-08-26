import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

import cv2
import numpy as np
from tqdm import tqdm


def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)


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


def batch_process_video(ref_video_path, dist_video_path, batch_size, step_size):
    ref_cap = cv2.VideoCapture(ref_video_path)
    dist_cap = cv2.VideoCapture(dist_video_path)
    ref_length = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dist_length = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_shape = (
        int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    dist_shape = (
        int(dist_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(dist_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    assert ref_length == dist_length, "Videos must have the same number of frames"
    assert ref_shape == dist_shape, "Videos must have the same resolution"

    psnr_values = []
    ssim_values = []

    current_step = 0

    with ThreadPoolExecutor() as executor:
        pbar = tqdm(total=ref_length // (batch_size * (step_size - 1)))
        while True:
            batch_ref_frames = []
            batch_dist_frames = []

            cur_size = 0
            while True:
                ret_ref, ref_frame = ref_cap.read()
                ret_dist, dist_frame = dist_cap.read()
                current_step += 1
                if (current_step % step_size) != 0:
                    continue
                if not ret_ref or not ret_dist:
                    break
                batch_ref_frames.append(ref_frame)
                batch_dist_frames.append(dist_frame)
                cur_size += 1
                if cur_size == batch_size:
                    break
                pass

            if not batch_ref_frames:
                pbar.update(1)
                break

            batch_psnr, batch_ssim = process_batch(
                executor, batch_ref_frames, batch_dist_frames
            )
            pbar.update(1)

            psnr_values.extend(batch_psnr)
            ssim_values.extend(batch_ssim)
            pass
        pbar.close()
        pass

    ref_cap.release()
    dist_cap.release()

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim


def main(
    reference_video: str,
    distorted_video: str,
    batch_size: int = 5000,
    step_size: int = 3,
) -> dict[str, Union[np.floating[Any], list[float]]]:
    avg_psnr, avg_ssim = batch_process_video(
        reference_video, distorted_video, batch_size=batch_size, step_size=step_size
    )

    # avg_vmaf = calculate_vmaf(reference_video, distorted_video)
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    # print(f"Average VMAF: {avg_vmaf}")
    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <reference_video> <distorted_video>")
        sys.exit(1)

    reference_video = sys.argv[1]
    distorted_video = sys.argv[2]
    main(reference_video, distorted_video)
