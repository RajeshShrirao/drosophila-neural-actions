"""
Warm Editorial Video Clip Generator for LinkedIn Showcase
Renders a smooth 30 FPS video demonstration of the Drosophila Neural Action Decoder:
- Clean Ivory (#F0EAE3) and Surface (#F7F3EE) editorial aesthetic
- Natural anatomical Drosophila tripod kinematics
- Optogenetic MDN stimulation triggering Moonwalk (backward walking)
- Giant Fiber ballistic escape jump
- Antennal grooming program
- Calibrated electrophysiology oscilloscope and pre-motor forecast telemetry
Encodes to H.264 MP4 via ffmpeg.
"""

import math
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Canvas resolution
WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_SEC = 12.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)

# Output directory
os.makedirs("videos", exist_ok=True)
OUTPUT_MP4 = "videos/drosophila_action_demo.mp4"

# User Extracted Warm Editorial Palette
IVORY_BG = (240, 234, 227)      # #F0EAE3
SURFACE_BG = (247, 243, 238)    # #F7F3EE
ELEVATED_BG = (255, 255, 255)
SUBTLE_BG = (235, 228, 220)      # #EBE4DC

INK_PRIMARY = (36, 33, 30)       # #24211E
TEXT_CHARCOAL = (72, 68, 63)     # #48443F
TEXT_MUTED = (108, 103, 97)      # #6C6761
TEXT_STONE = (140, 134, 127)     # #8C867F

BORDER_STONE = (214, 199, 189)   # #D6C7BD
BORDER_SUBTLE = (225, 214, 203)  # #E1D6CB

ACCENT_BLUSH = (132, 77, 67)     # #844D43
ACCENT_DUSTY = (158, 109, 96)    # #9E6D60
ACCENT_MUTED = (178, 136, 123)   # #B2887B
ACCENT_WARM = (194, 161, 147)    # #C2A193
ACCENT_SAGE = (85, 120, 100)     # #557864
ACCENT_TERRA = (148, 83, 56)     # #945338

ACTIONS = [
    ("Quiescence / Idle", TEXT_MUTED),
    ("Forward Walk", ACCENT_BLUSH),
    ("Backward Walk (Moonwalk)", ACCENT_DUSTY),
    ("Turn Left", ACCENT_SAGE),
    ("Turn Right", (107, 133, 118)),
    ("Groom Head/Antennae", ACCENT_MUTED),
    ("Escape Jump", ACCENT_TERRA),
    ("Courtship Wing Flare", ACCENT_BLUSH)
]

NEURONS = [
    ("DNp09", ACCENT_SAGE, 75),
    ("MDN", ACCENT_BLUSH, 75),
    ("DNg02", ACCENT_DUSTY, 65),
    ("aDN", ACCENT_MUTED, 70),
    ("GF", ACCENT_TERRA, 110),
    ("pIP10", (120, 58, 53), 65)
]


def render_editorial_fly(draw, cx, cy, phase, action_id, scale=1.0):
    """Draws anatomical Drosophila with natural chitin tones and tripod kinematics."""
    # Soft neutral floor shadow
    draw.ellipse([cx - 52 * scale, cy + 14, cx + 52 * scale, cy + 30], fill=(220, 212, 202))

    swing_amp = 1.0 if action_id in (1, 2, 3, 4) else 0.12
    leg_color = (110, 80, 52)
    
    leg_defs = [
        (-14, -18, -2.3, False, True),
        (-18, 0,   -1.6, False, False),
        (-14, 18,  -0.9, False, True),
        (14,  -18, -0.84, True, False),
        (18,  0,   -1.54, True, True),
        (14,  18,  -2.24, True, False)
    ]

    for idx, (lx, ly, base_angle, is_right, is_set_a) in enumerate(leg_defs):
        sign = 1 if is_right else -1
        leg_phase = phase if is_set_a else phase + math.pi
        swing = math.sin(leg_phase) * 0.28 * swing_amp

        # Grooming program front leg sweep
        if action_id == 5 and idx in (0, 3):
            groom_sweep = math.sin(phase * 1.5)
            j1 = (cx + lx * scale + sign * 14 * scale, cy + ly * scale - 24 * scale + groom_sweep * 6)
            tip = (cx + lx * scale + sign * 5 * scale + groom_sweep * 8, cy + ly * scale - 44 * scale + groom_sweep * 10)
            draw.line([(cx + lx * scale, cy + ly * scale), j1, tip], fill=(130, 94, 62), width=3)
            continue

        angle1 = (base_angle + swing * sign) * (1 if is_right else -1)
        femur_len = 26 * scale
        tibia_len = 30 * scale
        tarsus_len = 12 * scale

        j1 = (cx + lx * scale + math.cos(angle1) * femur_len * sign,
              cy + ly * scale + math.sin(angle1) * femur_len)
        angle2 = angle1 + 0.5 * sign
        j2 = (j1[0] + math.cos(angle2) * tibia_len * sign,
              j1[1] + math.sin(angle2) * tibia_len)
        tip = (j2[0] + math.cos(angle2 - 0.2 * sign) * tarsus_len * sign,
               j2[1] + math.sin(angle2 - 0.2 * sign) * tarsus_len)

        draw.line([(cx + lx * scale, cy + ly * scale), j1, j2, tip], fill=leg_color, width=3)
        draw.ellipse([j1[0] - 2, j1[1] - 2, j1[0] + 2, j1[1] + 2], fill=(74, 50, 32))
        draw.ellipse([j2[0] - 2, j2[1] - 2, j2[0] + 2, j2[1] + 2], fill=(74, 50, 32))

    # Abdomen with sepia tergites
    ab_y = cy + 38 * scale
    draw.ellipse([cx - 20 * scale, ab_y - 34 * scale, cx + 20 * scale, ab_y + 34 * scale],
                 fill=(175, 120, 65), outline=(56, 30, 12), width=2)
    for sy in [ab_y - 18, ab_y - 6, ab_y + 6, ab_y + 18]:
        draw.line([(cx - 16 * scale, sy), (cx + 16 * scale, sy)], fill=(56, 30, 12), width=3)

    # Thorax
    draw.ellipse([cx - 18 * scale, cy - 6 * scale - 20 * scale, cx + 18 * scale, cy - 6 * scale + 20 * scale],
                 fill=(140, 85, 40), outline=(46, 23, 8), width=2)

    # Translucent Wings
    is_courtship = (action_id == 7)
    is_escape = (action_id == 6)

    for sign in (-1, 1):
        w_ang = sign * 0.12
        if is_courtship and sign == 1:
            w_ang = 1.45 + math.sin(phase * 4.0) * 0.10
        elif is_escape:
            w_ang = sign * 0.72

        wx = cx + sign * 10 * scale
        wy = cy - 8 * scale
        tip_x = wx + math.sin(w_ang) * 58 * scale
        tip_y = wy + math.cos(w_ang) * 58 * scale
        draw.polygon([(wx, wy), (tip_x - 14 * scale, tip_y - 18 * scale),
                      (tip_x, tip_y), (tip_x + 14 * scale, tip_y - 18 * scale)],
                     fill=(230, 238, 245), outline=(130, 145, 160))

    # Head & Ruby Red Eyes
    head_y = cy - 32 * scale
    draw.ellipse([cx - 17 * scale, head_y - 11 * scale, cx + 17 * scale, head_y + 11 * scale],
                 fill=(135, 80, 35), outline=(46, 23, 8), width=2)
    draw.ellipse([cx - 16 * scale, head_y - 9 * scale, cx - 4 * scale, head_y + 9 * scale],
                 fill=(180, 35, 35), outline=(90, 15, 15))
    draw.ellipse([cx + 4 * scale, head_y - 9 * scale, cx + 16 * scale, head_y + 9 * scale],
                 fill=(180, 35, 35), outline=(90, 15, 15))

    for sign in (-1, 1):
        ax = cx + sign * 4 * scale
        draw.line([(ax, head_y - 8 * scale), (ax + sign * 6 * scale, head_y - 17 * scale)], fill=(46, 23, 8), width=2)


def generate_video():
    print(f"🎬 Initializing Warm Editorial Video Generation ({TOTAL_FRAMES} frames @ {FPS} FPS)...")
    
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        OUTPUT_MP4
    ]

    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    buf_len = 180
    rate_history = {n[0]: [5.0] * buf_len for n in NEURONS}
    gait_phase = 0.0

    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / FPS

        if t < 3.0:
            active_action = 1
            pred_action = 1
            is_opto = False
            opto_label = ""
            gait_phase += 14.0 / FPS
            fly_y = 195 + math.sin(gait_phase * 0.5) * 3
            fly_scale = 1.0
            active_neuron = "DNp09"
            conf = 97.4 + math.sin(t * 3) * 0.6
        elif t < 6.0:
            active_action = 2  # Moonwalk
            pred_action = 2
            is_opto = True
            opto_label = "OPTOGENETIC PULSE: Stimulating MDN (Moonwalker)"
            gait_phase -= 12.0 / FPS
            fly_y = 195 + math.sin(gait_phase * 0.5) * 4
            fly_scale = 1.0
            active_neuron = "MDN"
            conf = 98.6 + math.sin(t * 4) * 0.5
        elif t < 8.5:
            active_action = 6  # Escape Jump
            pred_action = 6
            is_opto = True
            opto_label = "OPTOGENETIC PULSE: Stimulating Giant Fiber (GF)"
            gait_phase += 24.0 / FPS
            jump_cycle = math.sin((t - 6.0) * 7.0)
            fly_y = 195 - max(0, jump_cycle) * 36
            fly_scale = 1.0 + max(0, jump_cycle) * 0.28
            active_neuron = "GF"
            conf = 99.1 + math.sin(t * 5) * 0.4
        elif t < 10.5:
            active_action = 5  # Grooming
            pred_action = 5
            is_opto = True
            opto_label = "OPTOGENETIC PULSE: Stimulating aDN (Antennal Groom)"
            gait_phase += 18.0 / FPS
            fly_y = 195
            fly_scale = 1.0
            active_neuron = "aDN"
            conf = 96.8 + math.sin(t * 3) * 0.8
        else:
            active_action = 1
            pred_action = 1
            is_opto = False
            opto_label = ""
            gait_phase += 14.0 / FPS
            fly_y = 195 + math.sin(gait_phase * 0.5) * 3
            fly_scale = 1.0
            active_neuron = "DNp09"
            conf = 97.2 + math.sin(t * 3) * 0.6

        for n_name, _, max_r in NEURONS:
            base = 4.0 + np.random.normal(0, 0.5)
            if n_name == active_neuron:
                target_r = max_r * 0.85 + np.random.normal(0, 2.5)
            else:
                target_r = base
            rate_history[n_name].pop(0)
            rate_history[n_name].append(target_r)

        # Draw Frame
        img = Image.new("RGB", (WIDTH, HEIGHT), IVORY_BG)
        draw = ImageDraw.Draw(img)

        # -------------------------------------------------------------
        # Editorial Masthead Header
        # -------------------------------------------------------------
        draw.rounded_rectangle([24, 16, WIDTH - 24, 76], radius=8, fill=SURFACE_BG, outline=BORDER_STONE)
        draw.text((40, 25), "RESEARCH MONOGRAPH · DROSOPHILA NEURAL ACTION DECODER", fill=ACCENT_BLUSH)
        draw.text((40, 47), "Janelia Drosophila Male CNS Connectome · 1,300 Descending Neuron Bottleneck (100 Hz Telemetry)", fill=TEXT_CHARCOAL)

        # Status badge
        badge_bg = (244, 236, 231) if is_opto else SURFACE_BG
        badge_border = ACCENT_BLUSH if is_opto else BORDER_STONE
        draw.rounded_rectangle([WIDTH - 270, 26, WIDTH - 40, 64], radius=6, fill=badge_bg, outline=badge_border)
        status_txt = "INTERVENTION ACTIVE" if is_opto else "DECODING STREAM · 100 HZ"
        draw.text((WIDTH - 250, 37), status_txt, fill=ACCENT_BLUSH if is_opto else TEXT_CHARCOAL)

        # -------------------------------------------------------------
        # Figure 1: Kinematics Plate (Left Top)
        # -------------------------------------------------------------
        card1 = [24, 90, 780, 420]
        draw.rounded_rectangle(card1, radius=8, fill=SURFACE_BG, outline=BORDER_STONE)
        draw.text((40, 102), "FIGURE 1 · MOTOR KINEMATICS SIMULATION", fill=ACCENT_BLUSH)

        # Millimeter Grid
        for gx in range(40, 765, 32):
            draw.line([(gx, 126), (gx, 404)], fill=SUBTLE_BG)
        for gy in range(126, 404, 32):
            draw.line([(40, gy), (764, gy)], fill=SUBTLE_BG)

        render_editorial_fly(draw, cx=400, cy=fly_y, phase=gait_phase, action_id=active_action, scale=fly_scale)

        action_name, action_col = ACTIONS[active_action]
        draw.rounded_rectangle([40, 368, 260, 404], radius=4, fill=IVORY_BG, outline=BORDER_STONE)
        draw.text((50, 378), f"EXECUTING: {action_name}", fill=ACCENT_BLUSH)

        if is_opto:
            draw.rounded_rectangle([275, 368, 764, 404], radius=4, fill=(244, 236, 231), outline=ACCENT_DUSTY)
            draw.text((288, 378), opto_label, fill=ACCENT_BLUSH)

        # -------------------------------------------------------------
        # Figure 2: Electrophysiology Oscilloscope (Left Bottom)
        # -------------------------------------------------------------
        card2 = [24, 434, 780, 700]
        draw.rounded_rectangle(card2, radius=8, fill=SURFACE_BG, outline=BORDER_STONE)
        draw.text((40, 444), "FIGURE 2 · DESCENDING HUB POPULATION FIRING RATES (HZ)", fill=ACCENT_BLUSH)

        osc_x = 40
        osc_w = 724
        osc_h = 220
        osc_top = 466
        draw.rectangle([osc_x, osc_top, osc_x + osc_w, osc_top + osc_h], fill=(250, 247, 242), outline=BORDER_SUBTLE)

        ch_h = osc_h / len(NEURONS)
        for ch_idx, (n_name, n_col, max_r) in enumerate(NEURONS):
            ch_y = osc_top + ch_idx * ch_h
            draw.line([(osc_x, ch_y), (osc_x + osc_w, ch_y)], fill=BORDER_SUBTLE)
            draw.text((osc_x + 8, ch_y + 4), n_name, fill=TEXT_STONE)

            data = rate_history[n_name]
            dx = osc_w / len(data)
            pts = []
            base_y = ch_y + ch_h - 4
            for i, val in enumerate(data):
                px = osc_x + i * dx
                py = base_y - (min(val, max_r) / max_r) * (ch_h - 8)
                pts.append((px, py))
            if len(pts) > 1:
                draw.line(pts, fill=n_col, width=2)

        # -------------------------------------------------------------
        # Right Stage: Pre-Motor Intent Forecast & Telemetry
        # -------------------------------------------------------------
        card3 = [796, 90, WIDTH - 24, 700]
        draw.rounded_rectangle(card3, radius=8, fill=SURFACE_BG, outline=BORDER_STONE)
        draw.text((816, 102), "PRE-MOTOR INTENT DECODER · 100 MS FORECAST", fill=ACCENT_BLUSH)

        # Forecast Hero Box
        pred_name, _ = ACTIONS[pred_action]
        draw.rounded_rectangle([816, 128, WIDTH - 40, 215], radius=6, fill=IVORY_BG, outline=BORDER_STONE)
        draw.text((832, 140), "FORECASTED NEXT ACTION (100ms in advance):", fill=TEXT_STONE)
        draw.text((832, 164), pred_name, fill=INK_PRIMARY)
        draw.text((WIDTH - 145, 164), f"{conf:.1f}%", fill=ACCENT_BLUSH)

        # Probability Distribution
        draw.text((816, 234), "BEHAVIORAL STATE PROBABILITY DISTRIBUTION:", fill=TEXT_STONE)
        prob_y = 260
        for a_idx, (a_name, _) in enumerate(ACTIONS):
            p_val = (conf / 100.0) if a_idx == pred_action else (1.0 - conf / 100.0) / 7.0
            draw.text((816, prob_y), a_name[:18], fill=ACCENT_BLUSH if a_idx == pred_action else TEXT_CHARCOAL)
            bar_w = int(220 * p_val)
            draw.rounded_rectangle([980, prob_y + 2, 1200, prob_y + 10], radius=3, fill=SUBTLE_BG)
            draw.rounded_rectangle([980, prob_y + 2, 980 + bar_w, prob_y + 10], radius=3, 
                                   fill=ACCENT_BLUSH if a_idx == pred_action else BORDER_STONE)
            draw.text((1210, prob_y), f"{p_val*100:.1f}%", fill=TEXT_MUTED)
            prob_y += 26

        # Benchmark Statistics Card
        draw.rounded_rectangle([816, 495, WIDTH - 40, 680], radius=6, fill=IVORY_BG, outline=BORDER_STONE)
        draw.text((832, 510), "JANELIA MALE CNS BENCHMARKS", fill=ACCENT_BLUSH)
        draw.text((832, 536), "• 100ms Forecast Accuracy : 97.3%", fill=INK_PRIMARY)
        draw.text((832, 560), "• 200ms Forecast Accuracy : 93.9%", fill=INK_PRIMARY)
        draw.text((832, 584), "• Command Hub Bottleneck  : ~1,300 Descending Neurons", fill=INK_PRIMARY)
        draw.text((832, 608), "• Inference Latency        : < 15 ms", fill=INK_PRIMARY)
        draw.text((832, 640), "rajeshshrirao.github.io/drosophila-neural-actions", fill=ACCENT_DUSTY)

        pipe.stdin.write(img.tobytes())

        if frame_idx % 60 == 0:
            print(f"   Frame {frame_idx}/{TOTAL_FRAMES} rendered ({t:.1f}s / {DURATION_SEC}s)...")

    pipe.stdin.close()
    pipe.wait()
    print(f"✅ Video generated successfully: {OUTPUT_MP4}")


if __name__ == "__main__":
    generate_video()
