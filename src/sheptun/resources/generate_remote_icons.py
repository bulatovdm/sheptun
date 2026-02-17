"""One-time script to generate remote status icons for menubar."""

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

RESOURCES = Path(__file__).parent
SIZES = {"": 50, "@2x": 100}
SUPERSAMPLE = 4  # Draw at 4x then downscale for anti-aliasing


def draw_arrow_up(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    """Draw a rounded upward arrow (↑) for remote send."""
    stroke = max(size // 6, 2)
    half_h = size  # Tall arrow with long shaft
    half_w = size * 3 // 8  # Compact chevron head

    # Chevron head (two rounded lines forming a V pointing up)
    head_y = cy - half_h // 2
    draw.line(
        [(cx - half_w, head_y + half_w), (cx, head_y)],
        fill="black",
        width=stroke,
        joint="curve",
    )
    draw.line(
        [(cx, head_y), (cx + half_w, head_y + half_w)],
        fill="black",
        width=stroke,
        joint="curve",
    )
    # Round caps at ends
    r = stroke // 2
    draw.ellipse([cx - half_w - r, head_y + half_w - r, cx - half_w + r, head_y + half_w + r], fill="black")
    draw.ellipse([cx + half_w - r, head_y + half_w - r, cx + half_w + r, head_y + half_w + r], fill="black")
    draw.ellipse([cx - r, head_y - r, cx + r, head_y + r], fill="black")

    # Shaft (vertical line with round caps)
    shaft_top = head_y
    shaft_bottom = cy + half_h // 2
    draw.line(
        [(cx, shaft_top), (cx, shaft_bottom)],
        fill="black",
        width=stroke,
    )
    draw.ellipse([cx - r, shaft_bottom - r, cx + r, shaft_bottom + r], fill="black")


def draw_arrow_down(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    """Draw a rounded downward arrow (↓) for remote receive."""
    stroke = max(size // 6, 2)
    half_h = size  # Tall arrow with long shaft
    half_w = size * 3 // 8  # Compact chevron head

    # Chevron head (two rounded lines forming a V pointing down)
    head_y = cy + half_h // 2
    draw.line(
        [(cx - half_w, head_y - half_w), (cx, head_y)],
        fill="black",
        width=stroke,
        joint="curve",
    )
    draw.line(
        [(cx, head_y), (cx + half_w, head_y - half_w)],
        fill="black",
        width=stroke,
        joint="curve",
    )
    # Round caps at ends
    r = stroke // 2
    draw.ellipse([cx - half_w - r, head_y - half_w - r, cx - half_w + r, head_y - half_w + r], fill="black")
    draw.ellipse([cx + half_w - r, head_y - half_w - r, cx + half_w + r, head_y - half_w + r], fill="black")
    draw.ellipse([cx - r, head_y - r, cx + r, head_y + r], fill="black")

    # Shaft (vertical line with round caps)
    shaft_top = cy - half_h // 2
    shaft_bottom = head_y
    draw.line(
        [(cx, shaft_top), (cx, shaft_bottom)],
        fill="black",
        width=stroke,
    )
    draw.ellipse([cx - r, shaft_top - r, cx + r, shaft_top + r], fill="black")


def generate_icon(
    base_name: str, output_name: str, arrow_fn: Any,
) -> None:
    for suffix, size in SIZES.items():
        base_path = RESOURCES / f"{base_name}{suffix}.png"
        base_img = Image.open(base_path).convert("RGBA")

        # Draw arrow at supersampled resolution for smooth edges
        ss = size * SUPERSAMPLE
        overlay_ss = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_ss)

        # Arrow in bottom-right corner, slightly larger
        arrow_size = int(size * 0.22 * SUPERSAMPLE)
        margin = int(size * 0.08 * SUPERSAMPLE)
        cx = ss - margin - arrow_size // 2
        cy = ss - margin - arrow_size // 2

        arrow_fn(draw, cx, cy, arrow_size)

        # Downscale with LANCZOS for anti-aliasing
        overlay = overlay_ss.resize((size, size), Image.Resampling.LANCZOS)

        result = Image.alpha_composite(base_img, overlay)
        out_path = RESOURCES / f"{output_name}{suffix}.png"
        result.save(out_path)
        print(f"  {out_path.name}")


def main() -> None:
    print("Generating remote send icons (↑):")
    generate_icon("mic_idle", "mic_remote_idle", draw_arrow_up)
    generate_icon("mic_active", "mic_remote_active", draw_arrow_up)

    print("Generating remote receive icons (↓):")
    generate_icon("mic_idle", "mic_receive", draw_arrow_down)
    generate_icon("mic_active", "mic_receive_active", draw_arrow_down)

    print("Done!")


if __name__ == "__main__":
    main()
