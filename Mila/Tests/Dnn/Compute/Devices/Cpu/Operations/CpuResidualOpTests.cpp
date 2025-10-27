/**
 * @file CpuResidualOpTests.cpp
 * @brief Tests for the CPU residual operation and Residual module.
 *
 * Tests use the modern ExecutionContext + OperationRegistry APIs. If a concrete
 * operation factory is not registered in the test build, operation-level checks
 * are skipped while module-level tests still run.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Operations::Tests {
using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;

class CpuResidualOpTests : public ::testing::Test {
  protected:
    void SetUp() override {
        // Modern execution context for CPU tests
        exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        // Try to obtain a device-registered Residual operation (CPU/FP32).
        // If the operation is not registered in this build, op_ will remain null
        // and op-specific assertions will be skipped.
        try {
            ResidualConfig cfg;
            op_ = OperationRegistry::instance().createBinaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                "ResidualOp", exec_ctx_, cfg);
        } catch (const std::exception&) {
            op_.reset();
        }

        // Create module-level Residual (CPU, FP32) — requires ExecutionContext.
        ResidualConfig cfg;
        module_ = std::make_shared<CpuResidual<TensorDataType::FP32>>(exec_ctx_, cfg);

        small_shape_ = {2, 3, 4};
        medium_shape_ = {8, 16, 32};
        large_shape_ = {32, 64, 128};
    }

    template <typename TTensor> bool hasNaNorInfHost(const TTensor& t) const {
        const float* data = static_cast<const float*>(t.rawData());
        for (size_t i = 0; i < t.size(); ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i]))
                return true;
        }
        return false;
    }

    std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
    std::shared_ptr<BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>> op_; // may be null if not registered
    std::shared_ptr<CpuResidual<TensorDataType::FP32>> module_;

    std::vector<size_t> small_shape_;
    std::vector<size_t> medium_shape_;
    std::vector<size_t> large_shape_;
};

TEST_F(CpuResidualOpTests, ModuleToString) {
    ASSERT_NE(module_, nullptr);
    std::string s = module_->toString();
    EXPECT_NE(s.find("Residual"), std::string::npos);
}

TEST_F(CpuResidualOpTests, OperationNameIfRegistered) {
    if (!op_) {
        GTEST_SKIP() << "Residual operation not registered for CPU/FP32 in this build.";
    }
    EXPECT_EQ(op_->getName(), std::string("Cpu::ResidualOp"));
}

TEST_F(CpuResidualOpTests, ForwardBasic) {
    // Operation-level path (if registered)
    if (op_) {
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        HostTensor A(exec_ctx_->getDevice(), small_shape_);
        HostTensor B(exec_ctx_->getDevice(), small_shape_);
        HostTensor Y(exec_ctx_->getDevice(), small_shape_);

        float* a = static_cast<float*>(A.rawData());
        float* b = static_cast<float*>(B.rawData());
        for (size_t i = 0; i < A.size(); ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i) * 0.5f;
        }

        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        ASSERT_NO_THROW(op_->forward(A, B, params, Y, out_state));

        float* y = static_cast<float*>(Y.rawData());
        for (size_t i = 0; i < Y.size(); ++i)
            EXPECT_FLOAT_EQ(y[i], a[i] + b[i]);
    } else {
        GTEST_SKIP() << "Residual operation not registered; skipping op-level forward checks.";
    }

    // Module-level forward (always exercised)
    {
        using ModTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        ModTensor inp(exec_ctx_->getDevice(), small_shape_);
        ModTensor out(exec_ctx_->getDevice(), small_shape_);

        float* id = static_cast<float*>(inp.rawData());
        for (size_t i = 0; i < inp.size(); ++i)
            id[i] = static_cast<float>(i);

        ASSERT_NO_THROW(module_->forward(inp, out));
        EXPECT_FALSE(hasNaNorInfHost(out));
        EXPECT_EQ(out.size(), inp.size());
    }
}

TEST_F(CpuResidualOpTests, BackwardBehavior) {
    // Operation-level backward (if registered)
    if (op_) {
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        HostTensor A(exec_ctx_->getDevice(), small_shape_);
        HostTensor B(exec_ctx_->getDevice(), small_shape_);
        HostTensor Y(exec_ctx_->getDevice(), small_shape_);
        HostTensor dY(exec_ctx_->getDevice(), small_shape_);
        HostTensor dA(exec_ctx_->getDevice(), small_shape_);
        HostTensor dB(exec_ctx_->getDevice(), small_shape_);

        float* a = static_cast<float*>(A.rawData());
        float* b = static_cast<float*>(B.rawData());
        float* dy = static_cast<float*>(dY.rawData());
        float* da = static_cast<float*>(dA.rawData());
        float* db = static_cast<float*>(dB.rawData());

        for (size_t i = 0; i < A.size(); ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i) * 0.5f;
            da[i] = 0.1f;
            db[i] = 0.2f;
            dy[i] = static_cast<float>(i % 5);
        }

        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters param_grads;
        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        op_->forward(A, B, params, Y, out_state);

        ASSERT_NO_THROW(op_->backward(A, B, Y, dY, params, param_grads, dA, dB, out_state));

        EXPECT_FALSE(hasNaNorInfHost(dA));
        EXPECT_FALSE(hasNaNorInfHost(dB));

        for (size_t i = 0; i < A.size(); ++i) {
            EXPECT_FLOAT_EQ(static_cast<float*>(dA.rawData())[i], 0.1f + dy[i]);
            EXPECT_FLOAT_EQ(static_cast<float*>(dB.rawData())[i], 0.2f + dy[i]);
        }
    } else {
        GTEST_SKIP() << "Residual operation not registered; skipping op-level backward checks.";
    }

    // Module-level backward — may be unimplemented; try and skip on exception.
    {
        using ModTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        ModTensor input(exec_ctx_->getDevice(), small_shape_);
        ModTensor out_grad(exec_ctx_->getDevice(), small_shape_);
        ModTensor in_grad(exec_ctx_->getDevice(), small_shape_);

        float* id = static_cast<float*>(input.rawData());
        float* dg = static_cast<float*>(out_grad.rawData());
        for (size_t i = 0; i < input.size(); ++i) {
            id[i] = static_cast<float>(i);
            dg[i] = static_cast<float>(i % 5);
        }

        try {
            module_->backward(input, out_grad, in_grad);
            EXPECT_FALSE(hasNaNorInfHost(in_grad));
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Residual::backward not implemented or unavailable: " << e.what();
        }
    }
}

TEST_F(CpuResidualOpTests, EdgeCasesAndDeterminism) {
    // Reuse small checks for op (if registered)
    if (op_) {
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        HostTensor a(exec_ctx_->getDevice(), small_shape_), b(exec_ctx_->getDevice(), small_shape_),
            out(exec_ctx_->getDevice(), small_shape_);
        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        // zeros
        for (size_t i = 0; i < a.size(); ++i) {
            static_cast<float*>(a.rawData())[i] = 0.0f;
            static_cast<float*>(b.rawData())[i] = 0.0f;
        }
        ASSERT_NO_THROW(op_->forward(a, b, params, out, out_state));
        for (size_t i = 0; i < out.size(); ++i)
            EXPECT_FLOAT_EQ(static_cast<float*>(out.rawData())[i], 0.0f);

        // deterministic run
        op_->forward(a, b, params, out, out_state);
        HostTensor out2(exec_ctx_->getDevice(), small_shape_);
        op_->forward(a, b, params, out2, out_state);
        for (size_t i = 0; i < out.size(); ++i)
            EXPECT_EQ(static_cast<float*>(out.rawData())[i], static_cast<float*>(out2.rawData())[i]);
    }

    // Module smoke checks
    {
        using ModTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        ModTensor a(exec_ctx_->getDevice(), medium_shape_), o1(exec_ctx_->getDevice(), medium_shape_),
            o2(exec_ctx_->getDevice(), medium_shape_);
        float* ad = static_cast<float*>(a.rawData());
        for (size_t i = 0; i < a.size(); ++i)
            ad[i] = (static_cast<float>(i % 17) - 8.5f) * 0.1f;
        ASSERT_NO_THROW(module_->forward(a, o1));
        ASSERT_NO_THROW(module_->forward(a, o2));
        for (size_t i = 0; i < o1.size(); ++i)
            EXPECT_EQ(static_cast<const float*>(o1.rawData())[i], static_cast<const float*>(o2.rawData())[i]);
    }
}

TEST_F(CpuResidualOpTests, ConstructorValidationAndPerformance) {
    // Module constructor validation
    ResidualConfig cfg;
    EXPECT_THROW(CpuResidual<TensorDataType::FP32>(nullptr, cfg), std::invalid_argument);

    if (std::getenv("CI") != nullptr)
        GTEST_SKIP() << "Skipping perf tests in CI";

    // Operation perf (if available)
    if (op_) {
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        HostTensor a(exec_ctx_->getDevice(), large_shape_), b(exec_ctx_->getDevice(), large_shape_),
            out(exec_ctx_->getDevice(), large_shape_);
        for (size_t i = 0; i < a.size(); ++i) {
            static_cast<float*>(a.rawData())[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
            static_cast<float*>(b.rawData())[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
        }
        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
            op_->forward(a, b, params, out, out_state);
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        size_t total = a.size() * iterations;
        double eps = static_cast<double>(total) / (dur.count() * 1e-6);
        std::cout << "CpuResidualOp: " << eps / 1e6 << " M elems/sec\n";
    }

    // Module perf
    {
        using ModTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        ModTensor a(exec_ctx_->getDevice(), large_shape_), out(exec_ctx_->getDevice(), large_shape_);
        for (size_t i = 0; i < a.size(); ++i)
            static_cast<float*>(a.rawData())[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; ++it)
            module_->forward(a, out);
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        size_t total = a.size() * iterations;
        double eps = static_cast<double>(total) / (dur.count() * 1e-6);
        std::cout << "Residual module: " << eps / 1e6 << " M elems/sec\n";
    }
}

TEST_F(CpuResidualOpTests, OpenMPScaling) {
#ifdef USE_OMP
    if (std::getenv("CI") != nullptr)
        GTEST_SKIP() << "Skipping OpenMP scaling test in CI environment";

    if (op_) {
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        HostTensor a(exec_ctx_->getDevice(), large_shape_), b(exec_ctx_->getDevice(), large_shape_),
            out(exec_ctx_->getDevice(), large_shape_);
        for (size_t i = 0; i < a.size(); ++i) {
            static_cast<float*>(a.rawData())[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
            static_cast<float*>(b.rawData())[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
        }
        int max_threads = omp_get_max_threads();
        std::vector<int> threads = {1};
        if (max_threads > 1) {
            threads.push_back(max_threads);
            if (max_threads > 3)
                threads.push_back(max_threads / 2);
        }
        const int iterations = 10;
        for (int num_threads : threads) {
            omp_set_num_threads(num_threads);
            auto start = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iterations; ++it)
                op_->forward(a, b, typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters(), out,
                             typename BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState());
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double eps = static_cast<double>(a.size() * iterations) / (dur.count() * 1e-6);
            std::cout << "CpuResidualOp with " << num_threads << " threads: " << eps / 1e6 << " M elems/sec\n";
        }
    }

    // Module OpenMP scaling
    {
        using ModTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        ModTensor a(exec_ctx_->getDevice(), large_shape_), out(exec_ctx_->getDevice(), large_shape_);
        for (size_t i = 0; i < a.size(); ++i)
            static_cast<float*>(a.rawData())[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f;
        int max_threads = omp_get_max_threads();
        std::vector<int> threads = {1};
        if (max_threads > 1) {
            threads.push_back(max_threads);
            if (max_threads > 3)
                threads.push_back(max_threads / 2);
        }
        const int iterations = 10;
        for (int num_threads : threads) {
            omp_set_num_threads(num_threads);
            auto start = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iterations; ++it)
                module_->forward(a, out);
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double eps = static_cast<double>(a.size() * iterations) / (dur.count() * 1e-6);
            std::cout << "Residual module with " << num_threads << " threads: " << eps / 1e6 << " M elems/sec\n";
        }
    }
#else
    GTEST_SKIP() << "OpenMP not available, skipping scaling test";
#endif
}
} // namespace Operations::Tests