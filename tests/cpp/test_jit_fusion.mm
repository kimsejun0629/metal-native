#import <gtest/gtest.h>
#include "metal_native/kernels/jit_fusion.h"
#include <string>

using namespace metal_native;

TEST(JITFusionTest, IsUnaryOp) {
    EXPECT_FALSE(is_unary_op(FusedOp::Add));
    EXPECT_FALSE(is_unary_op(FusedOp::Sub));
    EXPECT_FALSE(is_unary_op(FusedOp::Mul));
    EXPECT_FALSE(is_unary_op(FusedOp::Div));

    EXPECT_TRUE(is_unary_op(FusedOp::Exp));
    EXPECT_TRUE(is_unary_op(FusedOp::Log));
    EXPECT_TRUE(is_unary_op(FusedOp::Neg));
    EXPECT_TRUE(is_unary_op(FusedOp::Abs));
    EXPECT_TRUE(is_unary_op(FusedOp::Sqrt));
    EXPECT_TRUE(is_unary_op(FusedOp::ReLU));
    EXPECT_TRUE(is_unary_op(FusedOp::GELU));
    EXPECT_TRUE(is_unary_op(FusedOp::SiLU));
    EXPECT_TRUE(is_unary_op(FusedOp::Tanh));
    EXPECT_TRUE(is_unary_op(FusedOp::Sigmoid));
}

TEST(JITFusionTest, FusedOpName) {
    EXPECT_STREQ(fused_op_name(FusedOp::Add), "add");
    EXPECT_STREQ(fused_op_name(FusedOp::ReLU), "relu");
    EXPECT_STREQ(fused_op_name(FusedOp::GELU), "gelu");
}

TEST(JITFusionTest, FusedOpMSLExpr) {
    EXPECT_STREQ(fused_op_msl_expr(FusedOp::Add), "a + b");
    EXPECT_STREQ(fused_op_msl_expr(FusedOp::Exp), "exp(x)");
    EXPECT_STREQ(fused_op_msl_expr(FusedOp::ReLU), "max(x, VEC_ZERO)");
}

TEST(JITFusionTest, FusionKeyEquality) {
    FusionKey k1{{FusedOp::Add, FusedOp::ReLU}, FusedDType::Float32, 2};
    FusionKey k2{{FusedOp::Add, FusedOp::ReLU}, FusedDType::Float32, 2};
    FusionKey k3{{FusedOp::Add, FusedOp::Tanh}, FusedDType::Float32, 2};
    FusionKey k4{{FusedOp::Add, FusedOp::ReLU}, FusedDType::Float16, 2};

    EXPECT_TRUE(k1 == k2);
    EXPECT_FALSE(k1 == k3);
    EXPECT_FALSE(k1 == k4);
}

TEST(JITFusionTest, FusionKeyHash) {
    FusionKey k1{{FusedOp::Add, FusedOp::ReLU}, FusedDType::Float32, 2};
    FusionKey k2{{FusedOp::Add, FusedOp::ReLU}, FusedDType::Float32, 2};
    FusionKey k3{{FusedOp::Add, FusedOp::Tanh}, FusedDType::Float32, 2};

    FusionKeyHash hasher;
    EXPECT_EQ(hasher(k1), hasher(k2));
    EXPECT_NE(hasher(k1), hasher(k3));
}

TEST(JITFusionTest, SingletonAccess) {
    auto& compiler1 = JITFusionCompiler::instance();
    auto& compiler2 = JITFusionCompiler::instance();

    EXPECT_EQ(&compiler1, &compiler2);
}

TEST(JITFusionTest, EnabledByDefault) {
    auto& compiler = JITFusionCompiler::instance();
    EXPECT_TRUE(compiler.enabled());
}

TEST(JITFusionTest, SetEnabled) {
    auto& compiler = JITFusionCompiler::instance();

    compiler.set_enabled(false);
    EXPECT_FALSE(compiler.enabled());

    compiler.set_enabled(true);
    EXPECT_TRUE(compiler.enabled());
}

TEST(JITFusionTest, GenerateMSLUnaryChain) {
    auto& compiler = JITFusionCompiler::instance();

    // exp(relu(x))
    FusionKey key{{FusedOp::ReLU, FusedOp::Exp}, FusedDType::Float32, 1};
    std::string msl = compiler.generate_msl(key);

    EXPECT_FALSE(msl.empty());
    EXPECT_NE(msl.find("float4"), std::string::npos);
    EXPECT_NE(msl.find("input0"), std::string::npos);
    EXPECT_NE(msl.find("output"), std::string::npos);
    EXPECT_NE(msl.find("max(v, VEC_ZERO)"), std::string::npos);
    EXPECT_NE(msl.find("exp(v)"), std::string::npos);
}

TEST(JITFusionTest, GenerateMSLBinaryChain) {
    auto& compiler = JITFusionCompiler::instance();

    // (a + b) * c
    FusionKey key{{FusedOp::Add, FusedOp::Mul}, FusedDType::Float32, 3};
    std::string msl = compiler.generate_msl(key);

    EXPECT_FALSE(msl.empty());
    EXPECT_NE(msl.find("float4"), std::string::npos);
    EXPECT_NE(msl.find("input0"), std::string::npos);
    EXPECT_NE(msl.find("input1"), std::string::npos);
    EXPECT_NE(msl.find("input2"), std::string::npos);
    EXPECT_NE(msl.find("output"), std::string::npos);
}

TEST(JITFusionTest, GenerateMSLFloat16) {
    auto& compiler = JITFusionCompiler::instance();

    FusionKey key{{FusedOp::ReLU}, FusedDType::Float16, 1};
    std::string msl = compiler.generate_msl(key);

    EXPECT_FALSE(msl.empty());
    EXPECT_NE(msl.find("half4"), std::string::npos);
    EXPECT_EQ(msl.find("float4"), std::string::npos);  // Should not have float4
}

TEST(JITFusionTest, GenerateMSLMixedOps) {
    auto& compiler = JITFusionCompiler::instance();

    // gelu(a + b)
    FusionKey key{{FusedOp::Add, FusedOp::GELU}, FusedDType::Float32, 2};
    std::string msl = compiler.generate_msl(key);

    EXPECT_FALSE(msl.empty());
    EXPECT_NE(msl.find("input0"), std::string::npos);
    EXPECT_NE(msl.find("input1"), std::string::npos);
    EXPECT_NE(msl.find("VEC_GELU_CONST"), std::string::npos);
    EXPECT_NE(msl.find("VEC_GELU_COEFF"), std::string::npos);
}

TEST(JITFusionTest, EmptyOpsGeneratesEmpty) {
    auto& compiler = JITFusionCompiler::instance();

    FusionKey key{{}, FusedDType::Float32, 0};
    std::string msl = compiler.generate_msl(key);

    EXPECT_TRUE(msl.empty());
}

TEST(JITFusionTest, TooManyOpsGeneratesEmpty) {
    auto& compiler = JITFusionCompiler::instance();

    // 9 ops, exceeds MAX_CHAIN_LENGTH of 8
    std::vector<FusedOp> ops(9, FusedOp::ReLU);
    FusionKey key{ops, FusedDType::Float32, 1};
    std::string msl = compiler.generate_msl(key);

    EXPECT_TRUE(msl.empty());
}

TEST(JITFusionTest, CacheSizeInitiallyZero) {
    auto& compiler = JITFusionCompiler::instance();
    compiler.invalidate_all();

    EXPECT_EQ(compiler.cache_size(), 0);
}

TEST(JITFusionTest, HasCachedInitiallyFalse) {
    auto& compiler = JITFusionCompiler::instance();
    compiler.invalidate_all();

    FusionKey key{{FusedOp::ReLU}, FusedDType::Float32, 1};
    EXPECT_FALSE(compiler.has_cached(key));
}

TEST(JITFusionTest, GetPipelineReturnsNilWhenDisabled) {
    auto& compiler = JITFusionCompiler::instance();
    compiler.set_enabled(false);

    FusionKey key{{FusedOp::ReLU}, FusedDType::Float32, 1};
    auto pipeline = compiler.get_pipeline(key);

    EXPECT_EQ(pipeline, nullptr);

    compiler.set_enabled(true);
}

TEST(JITFusionTest, GetPipelineReturnsNilOnFirstCall) {
    auto& compiler = JITFusionCompiler::instance();
    compiler.invalidate_all();

    FusionKey key{{FusedOp::Add, FusedOp::ReLU}, FusedDType::Float32, 2};
    auto pipeline = compiler.get_pipeline(key);

    // First call should return nil and trigger async compilation
    EXPECT_EQ(pipeline, nullptr);

    // Should be marked as pending
    EXPECT_GT(compiler.pending_compilations(), 0);
}

TEST(JITFusionTest, InvalidateAllClearsCache) {
    auto& compiler = JITFusionCompiler::instance();

    // Trigger some compilations
    FusionKey k1{{FusedOp::ReLU}, FusedDType::Float32, 1};
    FusionKey k2{{FusedOp::Tanh}, FusedDType::Float32, 1};
    compiler.get_pipeline(k1);
    compiler.get_pipeline(k2);

    // Wait a moment for any compilations to complete
    usleep(100000);  // 100ms

    compiler.invalidate_all();
    EXPECT_EQ(compiler.cache_size(), 0);
}
