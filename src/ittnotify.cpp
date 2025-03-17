// #include <ittnotify.h>
// #include <pybind11/pybind11.h>

// namespace py = pybind11;

// // 封装 ITT API 的暂停和恢复采样功能
// void itt_pause() { __itt_pause(); }

// void itt_resume() { __itt_resume(); }

// // 让 VTune 在模块加载时自动暂停采样
// struct ITTController {
//   ITTController() {
//     __itt_pause(); // 加载模块时暂停 VTune 采样
//   }
// };
