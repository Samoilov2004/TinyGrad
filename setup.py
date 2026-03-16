import shutil
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


class CMakeBuildPy(build_py):
    def run(self):
        root = Path(__file__).resolve().parent
        build_dir = root / "build_python"

        build_dir.mkdir(parents=True, exist_ok=True)

        super().run()

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

        # Копируем библиотеку не в исходники, а в build output пакета
        pkg_build_dir = Path(self.build_lib) / "tinygradc"
        pkg_build_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(lib_path, pkg_build_dir / lib_path.name)


setup(
    name="tinygradc",
    version="0.1.0",
    description="Small C11 tensor/autograd runtime with a minimal Python wrapper",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    include_package_data=True,
    package_data={
        "tinygradc": ["*.so", "*.dylib", "*.dll"],
    },
    cmdclass={"build_py": CMakeBuildPy},
)
