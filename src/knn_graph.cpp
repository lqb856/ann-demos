#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

using MatrixXd = py::array_t<float, py::array::c_style | py::array::forcecast>;

struct Neighbor {
  float distance;
  int index;

  bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

py::list build_knn_graph(const MatrixXd &data, int knn_k, int n_threads = -1) {
  py::buffer_info buf = data.request();
  const int n = buf.shape[0];
  const int dim = buf.shape[1];
  const float *ptr = static_cast<const float *>(buf.ptr);

  if (n_threads > 0)
    omp_set_num_threads(n_threads);

  std::vector<std::vector<int>> graph(n);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    std::priority_queue<Neighbor> heap;

    const float *vec_i = ptr + i * dim;

    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;

      const float *vec_j = ptr + j * dim;
      float dist = 0.0;

      // 手动展开循环 + SIMD 优化
      for (int d = 0; d < dim; d += 4) {
        float diff0 = vec_i[d] - vec_j[d];
        float diff1 = (d + 1 < dim) ? vec_i[d + 1] - vec_j[d + 1] : 0.0;
        float diff2 = (d + 2 < dim) ? vec_i[d + 2] - vec_j[d + 2] : 0.0;
        float diff3 = (d + 3 < dim) ? vec_i[d + 3] - vec_j[d + 3] : 0.0;

        dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      }

      if (heap.size() < knn_k) {
        heap.push({dist, j});
      } else if (dist < heap.top().distance) {
        heap.pop();
        heap.push({dist, j});
      }
    }

    std::vector<int> neighbors;
    neighbors.reserve(knn_k);
    while (!heap.empty()) {
      neighbors.push_back(heap.top().index);
      heap.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end());
    graph[i] = std::move(neighbors);
  }

  // 转换为 Python 的 list
  py::list py_graph;
  for (const auto &neighbors : graph) {
    py::list py_neighbors;
    for (const auto &neighbor : neighbors) {
      py_neighbors.append(neighbor);
    }
    py_graph.append(py_neighbors);
  }

  return py_graph;
}

PYBIND11_MODULE(fast_knn, m) {
  m.def("build_knn_graph", &build_knn_graph, py::arg("data"), py::arg("knn_k"), py::arg("n_threads") = -1);
}
