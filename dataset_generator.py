"""
Fruit Dataset Generator
Generates a benchmark mini-dataset of fruit images (Apple, Banana, Orange)
organized in torchvision ImageFolder format (train/test per class).
Uses PIL with procedural textures, shapes, rotations, and background jitter.
"""

import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

IMG_SIZE = 64  # 64x64 RGB pixels


def create_background():
    """Generates a textured kitchen/table background."""
    bg_type = random.choice(["table", "plate", "neutral"])
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)

    if bg_type == "table":
        # Warm wood tone
        base_r = random.randint(180, 215)
        base_g = random.randint(140, 175)
        base_b = random.randint(100, 130)
        draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(base_r, base_g, base_b))
        # Subtle wood grain lines
        for y in range(0, IMG_SIZE, 3):
            line_shade = random.randint(-15, 15)
            draw.line([(0, y), (IMG_SIZE, y)], fill=(base_r + line_shade, base_g + line_shade, base_b + line_shade))
    elif bg_type == "plate":
        # White/soft ceramic
        base = random.randint(220, 245)
        draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(base, base, base))
    else:
        # Soft neutral gray/beige
        r = random.randint(200, 230)
        g = random.randint(200, 230)
        b = random.randint(195, 225)
        draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(r, g, b))

    # Add light noise
    arr = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw_apple():
    """Renders a procedural apple with shading, stem, and optional leaf."""
    bg = create_background()
    # Draw on a higher resolution layer then downsample for smooth anti-aliased edges
    scale = 2
    canvas_size = IMG_SIZE * scale
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cx = canvas_size // 2 + random.randint(-10, 10)
    cy = canvas_size // 2 + random.randint(-5, 10)
    radius_x = random.randint(32, 42)
    radius_y = int(radius_x * random.uniform(0.95, 1.05))

    # Apple lobes (left and right circles slightly overlapping)
    red_color = (random.randint(190, 235), random.randint(20, 45), random.randint(25, 45), 255)
    highlight_color = (random.randint(240, 255), random.randint(80, 120), random.randint(70, 100), 255)

    offset = int(radius_x * 0.22)
    draw.ellipse([cx - radius_x, cy - radius_y, cx + offset, cy + radius_y], fill=red_color)
    draw.ellipse([cx - offset, cy - radius_y, cx + radius_x, cy + radius_y], fill=red_color)
    draw.ellipse([cx - radius_x + 5, cy - radius_y + 12, cx + radius_x - 5, cy + radius_y - 2], fill=red_color)

    # Top stem depression (small darker patch)
    draw.ellipse([cx - 8, cy - radius_y - 2, cx + 8, cy - radius_y + 6], fill=(130, 20, 20, 255))

    # Brown stem
    stem_color = (101, 67, 33, 255)
    stem_top_x = cx + random.randint(-8, 8)
    stem_top_y = cy - radius_y - random.randint(12, 18)
    draw.line([(cx, cy - radius_y + 3), (stem_top_x, stem_top_y)], fill=stem_color, width=4)

    # Optional leaf
    if random.random() > 0.3:
        leaf_color = (34, 139, 34, 255)
        leaf_tip_x = stem_top_x + random.randint(8, 14)
        leaf_tip_y = stem_top_y + random.randint(2, 8)
        draw.polygon([(stem_top_x, stem_top_y + 5), (leaf_tip_x, leaf_tip_y), (stem_top_x + 6, stem_top_y + 10)], fill=leaf_color)

    # Soft highlight reflection
    draw.ellipse([cx - radius_x + 10, cy - radius_y + 14, cx - radius_x + 22, cy - radius_y + 30], fill=highlight_color)

    # Downsample and composite
    canvas = canvas.resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR)
    bg.paste(canvas, (0, 0), canvas)
    return bg


def draw_banana():
    """Renders a procedural curved yellow banana."""
    bg = create_background()
    scale = 2
    canvas_size = IMG_SIZE * scale
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cx = canvas_size // 2 + random.randint(-8, 8)
    cy = canvas_size // 2 + random.randint(-5, 8)
    length = random.randint(45, 55)
    arc_curve = random.uniform(0.5, 0.8)
    rot_angle = random.uniform(-40, 40)

    # Draw segmented banana spine
    yellow = (random.randint(235, 255), random.randint(195, 225), random.randint(20, 45), 255)
    green_tip = (random.randint(120, 160), random.randint(170, 200), random.randint(30, 60), 255)
    brown_tip = (random.randint(80, 110), random.randint(60, 80), random.randint(20, 40), 255)

    n_points = 25
    t_vals = np.linspace(-1, 1, n_points)
    points = []
    widths = []

    for t in t_vals:
        # Parabolic arc
        x = t * length
        y = arc_curve * (x ** 2) / length
        # Thickness profile
        w = max(4, int(16 * (1.0 - (t ** 2))))
        points.append((x, y))
        widths.append(w)

    # Rotate coordinates
    rad = math.radians(rot_angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot_pts = []
    for px, py in points:
        rx = cx + (px * cos_a - py * sin_a)
        ry = cy + (px * sin_a + py * cos_a)
        rot_pts.append((rx, ry))

    # Render overlapping circles along curve
    for i in range(len(rot_pts)):
        rx, ry = rot_pts[i]
        w = widths[i]
        # Endpoints color
        if i < 2:
            col = green_tip
        elif i == len(rot_pts) - 1:
            col = brown_tip
        else:
            col = yellow
        draw.ellipse([rx - w, ry - w, rx + w, ry + w], fill=col)

    # Downsample and composite
    canvas = canvas.resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR)
    bg.paste(canvas, (0, 0), canvas)
    return bg


def draw_orange():
    """Renders a procedural round orange with stippled texture and stem dot."""
    bg = create_background()
    scale = 2
    canvas_size = IMG_SIZE * scale
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cx = canvas_size // 2 + random.randint(-8, 8)
    cy = canvas_size // 2 + random.randint(-8, 8)
    r = random.randint(34, 42)

    orange_col = (random.randint(235, 255), random.randint(120, 150), random.randint(10, 25), 255)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=orange_col)

    # Porous texture (random small darker dots across surface)
    for _ in range(60):
        dot_dist = random.uniform(0, r - 4)
        dot_ang = random.uniform(0, 2 * math.pi)
        dx = int(cx + dot_dist * math.cos(dot_ang))
        dy = int(cy + dot_dist * math.sin(dot_ang))
        dot_col = (orange_col[0] - 18, orange_col[1] - 18, orange_col[2], 255)
        draw.point((dx, dy), fill=dot_col)

    # Small top calyx / stem button
    draw.ellipse([cx - 4, cy - r + 3, cx + 4, cy - r + 9], fill=(34, 110, 34, 255))
    draw.point((cx, cy - r + 6), fill=(70, 50, 20, 255))

    # Subtle bright highlight
    draw.ellipse([cx - r + 14, cy - r + 14, cx - r + 24, cy - r + 24], fill=(255, 180, 70, 180))

    canvas = canvas.resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR)
    bg.paste(canvas, (0, 0), canvas)
    return bg


def build_fruit_dataset(base_dir: str, n_train_per_class=100, n_test_per_class=30):
    generators = {
        "apple": draw_apple,
        "banana": draw_banana,
        "orange": draw_orange,
    }

    for split, count in [("train", n_train_per_class), ("test", n_test_per_class)]:
        for fruit_name, gen_func in generators.items():
            folder = os.path.join(base_dir, split, fruit_name)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                img = gen_func()
                img_path = os.path.join(folder, f"{fruit_name}_{i:04d}.png")
                img.save(img_path)

    total_train = n_train_per_class * len(generators)
    total_test = n_test_per_class * len(generators)
    print(f"Generated {total_train} train images and {total_test} test images in: {base_dir}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")
    build_fruit_dataset(dataset_dir, n_train_per_class=120, n_test_per_class=35)
