# setup.py
from setuptools import setup, Extension
import pybind11
import os

# 获取当前目录路径
base_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(base_dir, "python")

setup(
    name="fast_knn",
    ext_modules=[
        Extension(
            "fast_knn",
            ["src/knn_graph.cpp"],
            include_dirs=[pybind11.get_include()],
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