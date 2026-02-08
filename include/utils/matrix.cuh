#ifndef FASTTRACKER_MATRIX_CUH
#define FASTTRACKER_MATRIX_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace fasttracker {
namespace cuda {

// デバイス関数: 行列乗算 C = A * B
// A: [m x k], B: [k x n], C: [m x n]
__device__ inline void matmul(const float* A, const float* B, float* C,
                               int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// デバイス関数: 行列転置 B = A^T
// A: [m x n], B: [n x m]
__device__ inline void transpose(const float* A, float* B, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

// デバイス関数: 行列加算 C = A + B
// A, B, C: [m x n]
__device__ inline void matadd(const float* A, const float* B, float* C,
                               int m, int n) {
    int size = m * n;
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// デバイス関数: 行列減算 C = A - B
// A, B, C: [m x n]
__device__ inline void matsub(const float* A, const float* B, float* C,
                               int m, int n) {
    int size = m * n;
    for (int i = 0; i < size; i++) {
        C[i] = A[i] - B[i];
    }
}

// デバイス関数: ベクトルの内積
__device__ inline float dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// デバイス関数: ベクトルのノルム
__device__ inline float norm(const float* a, int n) {
    return sqrtf(dot(a, a, n));
}

// デバイス関数: Cholesky分解（下三角行列）
// A: [n x n] 正定値対称行列, L: [n x n] 下三角行列
// A = L * L^T
__device__ inline bool cholesky(const float* A, float* L, int n) {
    // Lをゼロ初期化
    for (int i = 0; i < n * n; i++) {
        L[i] = 0.0f;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;

            if (i == j) {
                for (int k = 0; k < j; k++) {
                    sum += L[j * n + k] * L[j * n + k];
                }
                float val = A[j * n + j] - sum;
                if (val <= 0.0f) {
                    return false; // 正定値でない
                }
                L[j * n + j] = sqrtf(val);
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return true;
}

// デバイス関数: 行列の逆行列（小サイズ行列用、Gauss-Jordan法）
// A: [n x n], Ainv: [n x n]
__device__ inline bool invert(const float* A, float* Ainv, int n) {
    // 拡大行列 [A | I] を作成
    float aug[6 * 12]; // 最大6x6行列を想定
    if (n > 6) return false;

    // [A | I] の初期化
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * 2 * n + j] = A[i * n + j];
            aug[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Gauss-Jordan消去法
    for (int i = 0; i < n; i++) {
        // ピボット選択
        float pivot = aug[i * 2 * n + i];
        if (fabsf(pivot) < 1e-10f) {
            return false; // 特異行列
        }

        // i行目を正規化
        for (int j = 0; j < 2 * n; j++) {
            aug[i * 2 * n + j] /= pivot;
        }

        // 他の行を消去
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = aug[k * 2 * n + i];
                for (int j = 0; j < 2 * n; j++) {
                    aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                }
            }
        }
    }

    // 逆行列を取り出す
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ainv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    return true;
}

// デバイス関数: 行列式（小サイズ行列用）
__device__ inline float determinant(const float* A, int n) {
    if (n == 1) {
        return A[0];
    } else if (n == 2) {
        return A[0] * A[3] - A[1] * A[2];
    } else if (n == 3) {
        return A[0] * (A[4] * A[8] - A[5] * A[7])
             - A[1] * (A[3] * A[8] - A[5] * A[6])
             + A[2] * (A[3] * A[7] - A[4] * A[6]);
    }
    // より大きなサイズは未実装
    return 0.0f;
}

// デバイス関数: 対称行列の固有値分解（簡易版、2x2のみ）
__device__ inline void eigen2x2(const float* A, float* eigenvalues, float* eigenvectors) {
    // A = [a b]
    //     [b c]
    float a = A[0];
    float b = A[1];
    float c = A[3];

    float trace = a + c;
    float det = a * c - b * b;
    float discriminant = sqrtf(trace * trace - 4.0f * det);

    eigenvalues[0] = 0.5f * (trace + discriminant);
    eigenvalues[1] = 0.5f * (trace - discriminant);

    // 固有ベクトル（正規化されていない）
    if (fabsf(b) > 1e-10f) {
        eigenvectors[0] = eigenvalues[0] - c;
        eigenvectors[1] = b;
        eigenvectors[2] = eigenvalues[1] - c;
        eigenvectors[3] = b;

        // 正規化
        float norm1 = sqrtf(eigenvectors[0] * eigenvectors[0] + eigenvectors[1] * eigenvectors[1]);
        float norm2 = sqrtf(eigenvectors[2] * eigenvectors[2] + eigenvectors[3] * eigenvectors[3]);
        eigenvectors[0] /= norm1;
        eigenvectors[1] /= norm1;
        eigenvectors[2] /= norm2;
        eigenvectors[3] /= norm2;
    } else {
        // 対角行列
        eigenvectors[0] = 1.0f;
        eigenvectors[1] = 0.0f;
        eigenvectors[2] = 0.0f;
        eigenvectors[3] = 1.0f;
    }
}

// デバイス関数: 重み付き平均
__device__ inline void weighted_mean(const float* points, const float* weights,
                                      float* mean, int num_points, int dim) {
    for (int i = 0; i < dim; i++) {
        mean[i] = 0.0f;
    }

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dim; j++) {
            mean[j] += weights[i] * points[i * dim + j];
        }
    }
}

// デバイス関数: 外積（2つのベクトルの外積行列）
// a: [n], b: [m], C: [n x m]
// C = a * b^T
__device__ inline void outer(const float* a, const float* b, float* C,
                              int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = a[i] * b[j];
        }
    }
}

} // namespace cuda
} // namespace fasttracker

#endif // FASTTRACKER_MATRIX_CUH
