import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

from sheptun import __version__

ICON_PADDING_RATIO = 0.70
ICON_SIZES = [16, 32, 64, 128, 256, 512]
MAX_RETINA_SIZE = 1024


def write_info_plist(path: Path) -> None:
    path.write_text(f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>sheptun</string>
    <key>CFBundleIdentifier</key>
    <string>com.sheptun.menubar</string>
    <key>CFBundleName</key>
    <string>Sheptun</string>
    <key>CFBundleDisplayName</key>
    <string>Sheptun</string>
    <key>CFBundleVersion</key>
    <string>{__version__}</string>
    <key>CFBundleShortVersionString</key>
    <string>{__version__}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Sheptun needs microphone access for voice recognition</string>
</dict>
</plist>
""")


def write_executable(path: Path, project_dir: Path, venv_dir: Path) -> None:
    path.write_text(f"""\
#!/bin/bash
cd "{project_dir}"
source "{venv_dir}/bin/activate"
exec python -m sheptun.menubar
""")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _save_icon(img: "PILImage.Image", path: Path, size: int) -> None:
    from PIL import Image

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon_size = int(size * ICON_PADDING_RATIO)
    offset = (size - icon_size) // 2
    resized = img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
    canvas.paste(resized, (offset, offset), resized)
    canvas.save(path)


def generate_app_icons(icon_src: Path, iconset_dir: Path, output_icns: Path) -> None:
    from PIL import Image

    if not icon_src.exists():
        return

    img = Image.open(icon_src).convert("RGBA")

    for size in ICON_SIZES:
        _save_icon(img, iconset_dir / f"icon_{size}x{size}.png", size)
        retina_size = size * 2
        if retina_size <= MAX_RETINA_SIZE:
            _save_icon(img, iconset_dir / f"icon_{size}x{size}@2x.png", retina_size)

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(output_icns)],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(iconset_dir)


def build_app(app_dir: Path) -> None:
    project_dir = Path(__file__).parent.parent.parent
    venv_dir = Path(sys.executable).parent.parent
    resources_dir = Path(__file__).parent / "resources"

    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_app_dir = contents_dir / "Resources"
    iconset_dir = resources_app_dir / "AppIcon.iconset"

    if app_dir.exists():
        shutil.rmtree(app_dir)

    macos_dir.mkdir(parents=True)
    iconset_dir.mkdir(parents=True)

    write_info_plist(contents_dir / "Info.plist")
    write_executable(macos_dir / "sheptun", project_dir, venv_dir)
    generate_app_icons(
        resources_dir / "microphone-idle.png",
        iconset_dir,
        resources_app_dir / "AppIcon.icns",
    )
