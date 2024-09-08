import re
import subprocess
import sys
from typing import Any, Union

import numpy as np
from tqdm import tqdm


def calculate_psnr(ref_path, dist_path) -> str:
    # Get the total number of frames in the video for the progress bar
    probe_result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            ref_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    total_frames = int(probe_result.stdout.strip())
    # Run the ffmpeg command with Popen to capture stdout in real-time
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            dist_path,
            "-ss",
            # "0.516666666",  # tooth2
            # "0.5",  # webrtc+2
            # "0.5833333333333334",  # nofec
            # "0.5833333333333334",  # RL
            # "0.6339666666666667",  # webrtc
            # "0.55",  # tooth_oldgs
            # "0.5666666666666667",  # nofec_oldgs
            # "0.7333333333333333",  # webrtc+_oldgs
            "0.8666666666666667",  # rl_oldgs
            "-i",
            ref_path,
            "-lavfi",
            "psnr=stats_file=/dev/null",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        universal_newlines=True,
    )

    psnr_score = ""
    frame_regex = re.compile(r"frame=\s*(\d+)")  # Regex to capture frame number

    with tqdm(total=total_frames, desc="Processing PSNR", unit="frame") as pbar:
        assert process.stderr is not None
        for line in process.stderr:
            # Capture PSNR score from the output
            if " PSNR y:" in line:
                psnr_score = line.split(" PSNR ")[1]
                pbar.update(total_frames - pbar.n)
                break
            # Update progress bar based on frame number
            elif "frame=" in line:
                match = frame_regex.search(line)
                if match:
                    current_frame = int(match.group(1))
                    pbar.update(current_frame - pbar.n)
                    pass
            else:
                # print(line, end="")
                pass
            pass
        pass

    process.kill()
    pbar.close()

    return psnr_score


def calculate_vmaf(ref_path, dist_path):
    # Get the total number of frames in the video for the progress bar
    probe_result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            ref_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    total_frames = int(probe_result.stdout.strip())
    # Run the ffmpeg command with Popen to capture stdout in real-time
    process = subprocess.Popen(
        [
            "/home/kjy/bin/ffmpeg",
            # "-y",
            # "-stats",
            # "-loglevel",
            # "info",
            "-i",
            dist_path,
            "-ss",
            "0.516666666",  # tooth2
            # "0.5",  # webrtc+2
            # "0.5833333333333334",  # nofec
            # "0.5833333333333334",  # RL
            # "0.6339666666666667",  # webrtc
            # "0.55",  # tooth_oldgs
            # "0.5666666666666667",  # nofec_oldgs
            # "0.7333333333333333",  # webrtc+_oldgs
            # "0.8666666666666667",  # rl_oldgs
            "-i",
            ref_path,
            "-lavfi",
            f"[0:v]setpts=PTS-STARTPTS[reference];[1:v]setpts=PTS-STARTPTS[distorted];[distorted][reference]libvmaf=log_fmt=csv:log_path=output.csv:model=path={'/home/kjy/Documents/projects/vmaf'}/model/vmaf_v0.6.1.json:n_threads=8:n_subsample=5:feature=name=psnr|name=float_ssim",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        universal_newlines=True,
    )

    vmaf_scores = []
    frame_regex = re.compile(r"frame=\s*(\d+)")  # Regex to capture frame number

    with tqdm(total=total_frames, desc="Processing VMAF", unit="frame") as pbar:
        assert process.stderr is not None
        for line in process.stderr:
            # print(line, end="")
            # Capture VMAF score from the output
            if "VMAF score" in line:
                score = float(line.split()[-1])
                vmaf_scores.append(score)
                pbar.update(total_frames - pbar.n)
                break
            # Update progress bar based on frame number
            elif "frame=" in line:
                match = frame_regex.search(line)
                if match:
                    current_frame = int(match.group(1))
                    pbar.update(current_frame - pbar.n)
                    pass
            else:
                # print(line, end="")
                pass
            pass
        process.stderr.close()
        if process.stdout:
            process.stdout.close()
        pass

    process.wait()
    pbar.close()
    return np.mean(vmaf_scores)


def main(
    reference_video: str,
    distorted_video: str,
) -> dict[str, Union[np.floating[Any], list[float]]]:
    # psnr_score = calculate_psnr(reference_video, distorted_video)
    # print(psnr_score)

    avg_vmaf = calculate_vmaf(reference_video, distorted_video)
    return {
        # "psnr": avg_psnr,
        # "ssim": avg_ssim,
        "vmaf": avg_vmaf,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <reference_video> <distorted_video>")
        sys.exit(1)

    reference_video = sys.argv[1]
    distorted_video = sys.argv[2]
    res = main(reference_video, distorted_video)
    # print(f"Average PSNR: {res['psnr']}")
    # print(f"Average SSIM: {res['ssim']}")
    print(f"Average VMAF: {res['vmaf']}")
