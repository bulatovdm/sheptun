"""One-time script to generate remote status icons for menubar."""

from pathlib import Path

from PIL import Image, ImageDraw

RESOURCES = Path(__file__).parent
SIZES = {"": 50, "@2x": 100}


def draw_arrow_up(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    """Draw an upward arrow (↑) for remote send."""
    half = size // 2
    tip_y = cy - half
    base_y = cy + half
    # Arrow head
    draw.polygon(
        [(cx, tip_y), (cx - half, cy - half // 3), (cx + half, cy - half // 3)],
        fill="black",
    )
    # Arrow shaft
    shaft_w = size // 4
    draw.rectangle(
        [cx - shaft_w, cy - half // 3, cx + shaft_w, base_y],
        fill="black",
    )


def draw_arrow_down(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    """Draw a downward arrow (↓) for remote receive."""
    half = size // 2
    tip_y = cy + half
    base_y = cy - half
    # Arrow head
    draw.polygon(
        [(cx, tip_y), (cx - half, cy + half // 3), (cx + half, cy + half // 3)],
        fill="black",
    )
    # Arrow shaft
    shaft_w = size // 4
    draw.rectangle(
        [cx - shaft_w, base_y, cx + shaft_w, cy + half // 3],
        fill="black",
    )


def generate_icon(
    base_name: str, output_name: str, arrow_fn: type(draw_arrow_up)
) -> None:
    for suffix, size in SIZES.items():
        base_path = RESOURCES / f"{base_name}{suffix}.png"
        base_img = Image.open(base_path).convert("RGBA")

        overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Arrow in bottom-right corner
        arrow_size = size // 5
        margin = size // 10
        cx = size - margin - arrow_size // 2
        cy = size - margin - arrow_size // 2

        arrow_fn(draw, cx, cy, arrow_size)

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
