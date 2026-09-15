"""
Convert Browser Subagent WebP Recording into High-Definition MP4 for LinkedIn
Extracts frames from the 1920x986 browser subagent session recording and encodes to H.264 MP4.
"""

import os
import subprocess
from PIL import Image, ImageSequence

WEBP_PATH = "/Users/rajeshshrirao/.gemini/antigravity-ide/brain/b4f30df4-ad2f-40bd-a204-6633bb5e9ff6/fly_editorial_demo_1789476623270.webp"
OUTPUT_DIR = "/Users/rajeshshrirao/Desktop/Flybrain/videos"
OUTPUT_MP4 = os.path.join(OUTPUT_DIR, "drosophila_action_demo.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🎬 Opening browser recording: {WEBP_PATH}")
im = Image.open(WEBP_PATH)
n_frames = getattr(im, "n_frames", 1)
print(f"   Resolution: {im.size}, Frames: {n_frames}")

# Extract frames and durations
frames = []
durations = []
for frame in ImageSequence.Iterator(im):
    # Ensure RGB
    rgb_frame = frame.convert("RGB")
    frames.append(rgb_frame)
    dur = frame.info.get("duration", 250)  # default 250ms if not specified
    durations.append(dur)

print(f"   Extracted {len(frames)} frames. Average duration: {sum(durations)/len(durations):.1f} ms")

# Ensure dimensions are even numbers for H.264 (1920x986 is even!)
w, h = frames[0].size
if w % 2 != 0: w -= 1
if h % 2 != 0: h -= 1

# Encode to MP4 using ffmpeg at 20 FPS (repeating frames based on their actual duration)
target_fps = 20
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{w}x{h}",
    "-pix_fmt", "rgb24",
    "-r", str(target_fps),
    "-i", "-",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "medium",
    "-crf", "18",
    "-movflags", "+faststart",
    OUTPUT_MP4
]

pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

total_written = 0
for idx, (frame, dur) in enumerate(zip(frames, durations)):
    # Calculate how many video frames this WebP frame should span
    num_repeats = max(1, int(round((dur / 1000.0) * target_fps)))
    # Resize slightly if odd
    if frame.size != (w, h):
        frame = frame.resize((w, h), Image.Resampling.LANCZOS)
    raw_bytes = frame.tobytes()
    for _ in range(num_repeats):
        pipe.stdin.write(raw_bytes)
        total_written += 1

pipe.stdin.close()
pipe.wait()

print(f"✅ Successfully converted browser session to MP4!")
print(f"   Output: {OUTPUT_MP4}")
print(f"   Total video frames written: {total_written} (~{total_written/target_fps:.1f} seconds)")
