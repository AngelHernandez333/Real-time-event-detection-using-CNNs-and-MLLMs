#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

double sum_cpp(pybind11::array_t<double> arr) {
    auto buf = arr.unchecked();
    double total = 0;
    for (ssize_t i = 0; i < buf.shape(0); i++) {
        total += buf(i);
    }
    return total;
}

PYBIND11_MODULE(example, m) {
    m.def("sum_cpp", &sum_cpp, "Sum elements of an array");
}
