# setup.py
from setuptools import setup, Extension
import pybind11
import os

# 获取当前目录路径
base_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(base_dir, "python")

vtune_lib_path = "/opt/intel/oneapi/vtune/latest/sdk/lib64"

setup(
    name="fast_knn",
    ext_modules=[
        Extension(
            "fast_knn",
            ["src/knn_graph.cpp",
             "src/ittnotify.cpp",],
            include_dirs=[pybind11.get_include(), "/opt/intel/oneapi/vtune/latest/sdk/include"],
            library_dirs=[vtune_lib_path],  # 指定库路径
            libraries=["ittnotify", "jitprofiling"],  # 链接 libittnotify.a
            extra_compile_args=["-std=c++17", "-O3", "-fopenmp", "-march=native"],
            extra_link_args=["-fopenmp"],
            language="c++"
        )
    ],
    # 关键配置：指定包目录
    package_dir={"": "python"},  
    packages=[""], 
    # 包含编译后的so文件
    package_data={"": ["*.so"]}, 
)