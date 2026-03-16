import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class CMakeBuildPy(build_py):
    def run(self):
        root = Path(__file__).resolve().parent
        build_dir = root / "build_python"
        pkg_dir = root / "python" / "tinygradc"

        build_dir.mkdir(parents=True, exist_ok=True)
        pkg_dir.mkdir(parents=True, exist_ok=True)

        subprocess.check_call([
            "cmake",
            "-S", str(root),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ])

        subprocess.check_call([
            "cmake",
            "--build", str(build_dir),
            "--config", "Release",
        ])

        lib_candidates = list(build_dir.rglob("libtinygradc.so"))
        lib_candidates += list(build_dir.rglob("libtinygradc.dylib"))
        lib_candidates += list(build_dir.rglob("tinygradc.dll"))

        if not lib_candidates:
            raise RuntimeError("Could not find built tinygradc shared library")

        lib_path = lib_candidates[0]
        shutil.copy2(lib_path, pkg_dir / lib_path.name)

        super().run()


setup(
    cmdclass={"build_py": CMakeBuildPy},
)
