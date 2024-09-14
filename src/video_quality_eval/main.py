import re
import subprocess
import sys
from typing import Any, Union

import numpy as np
from tqdm import tqdm


def get_video_resolution(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    width, height = map(int, result.stdout.strip().split("x"))
    return width, height


def calculate_vmaf(ref_path, dist_path):
    # Get resolutions
    ref_width, ref_height = get_video_resolution(ref_path)
    dist_width, dist_height = get_video_resolution(dist_path)

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

    # Prepare ffmpeg command
    ffmpeg_command = [
        "/home/kjy/bin/ffmpeg",
        "-i",
        dist_path,
        # "-ss",
        # "0.516666666",  # tooth2
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
    ]

    vmaf_filter = "[distorted][reference]libvmaf=log_fmt=csv:log_path=output.csv:model=path={'/home/kjy/Documents/projects/vmaf'}/model/vmaf_v0.6.1.json:n_threads=8:n_subsample=5:feature=name=psnr|name=float_ssim"

    # Add scaling if necessary
    if dist_width < ref_width or dist_height < ref_height:
        ffmpeg_command.extend(
            [
                "-filter_complex",
                f"[0:v]scale={ref_width}:{ref_height}:flags=bicubic[scaled];[scaled]setpts=PTS-STARTPTS[distorted];[1:v]setpts=PTS-STARTPTS[reference];{vmaf_filter}",
            ]
        )
    else:
        ffmpeg_command.extend(
            [
                "-lavfi",
                f"[0:v]setpts=PTS-STARTPTS[distorted];[1:v]setpts=PTS-STARTPTS[reference];{vmaf_filter}",
            ]
        )

    ffmpeg_command.extend(["-f", "null", "-"])

    # Run the ffmpeg command with Popen to capture stdout in real-time
    process = subprocess.Popen(
        ffmpeg_command,
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
        process.stderr.close()
        if process.stdout:
            process.stdout.close()

    process.wait()
    pbar.close()
    return np.mean(vmaf_scores)


def main(
    reference_video: str,
    distorted_video: str,
) -> dict[str, Union[np.floating[Any], list[float]]]:
    avg_vmaf = calculate_vmaf(reference_video, distorted_video)
    return {
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
