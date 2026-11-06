import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


root = Path(SPECPATH)
assets = [(str(root / filename), ".") for filename in ("demo.html", "maplescreen_alerts.js")]
assets += collect_data_files("rapidocr_onnxruntime")
assets.append((str(root / "LICENSE"), "licenses"))

runtime = Path(sys.base_prefix) / "Library" / "bin"
runtime_dlls = ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll", "sqlite3.dll")
binaries = collect_dynamic_libs("onnxruntime")
binaries += [(str(runtime / name), ".") for name in runtime_dlls if (runtime / name).is_file()]

analysis = Analysis(
    [str(root / "monitor_chat.py")],
    pathex=[str(root / ".deps")],
    binaries=binaries,
    datas=assets,
)
archive = PYZ(analysis.pure)
exe = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    name="MapleScreen",
    console=True,
    upx=False,
)
