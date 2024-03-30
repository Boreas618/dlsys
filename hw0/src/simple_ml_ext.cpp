#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul_2d(const float *x, const float *y, const size_t r1, const size_t c1,
               const size_t r2, const size_t c2, float *z) {
  for (size_t i = 0; i < r1; i++) {
    for (size_t j = 0; j < c2; j++) {
      z[i * c2 + j] = 0;
      for (size_t k = 0; k < c1; k++) {
        z[i * c2 + j] += x[i * c1 + k] * y[k * c2 + j];
      }
    }
  }
}

void transpose(const float *x, size_t r, size_t c, float *x_T) {
  for (size_t i = 0; i < r; i++) {
    for (size_t j = 0; j < c; j++) {
      x_T[j * r + i] = x[i * c + j];
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  const size_t iterations = m / batch;
  for (size_t it = 0; it < iterations; it++) {
    const float *x_ = X + it * batch * n;
    const size_t m_ = std::min(batch, m - it * batch);
    const unsigned char *y_ = y + it * batch;
    float *logits = (float *)malloc(m_ * k * sizeof(float));

    matmul_2d(x_, theta, m_, n, n, k, logits);
    for (size_t i = 0; i < m_ * k; i++) *(logits + i) = exp(*(logits + i));

    for (size_t i = 0; i < m_; i++) {
      float sum = 0;
      for (size_t j = 0; j < k; j++) sum += logits[i * k + j];
      for (size_t j = 0; j < k; j++) logits[i * k + j] /= sum;
    }

    for (size_t i = 0; i < m_; i++) logits[i * k + y_[i]] -= 1;

    float *x_T = (float *)malloc(m_ * n * sizeof(float));
    transpose(x_, m_, n, x_T);

    float *grad = (float *)malloc(k * n * sizeof(float));
    std::cout << grad[0] << std::endl;
    matmul_2d(x_T, logits, n, m_, m_, k, grad);
    std::cout << x_T[0] << std::endl;

    for (size_t j = 0; j < n; j++)
      for (size_t l = 0; l < k; l++)
        theta[j * k + l] -= grad[j * k + l] * lr / m_;

    free(logits);
    free(x_T);
    free(grad);
  }
  return;
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
