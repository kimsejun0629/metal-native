/// @file test_error.cpp
/// @brief Unit tests for error handling (MNException, MN_CHECK, MN_THROW).

#include <gtest/gtest.h>
#include "metal_native/core/error.h"

#include <string>

using namespace metal_native;

// -- error_to_string tests ----------------------------------------------------

TEST(ErrorTest, ErrorToStringSuccess) {
    EXPECT_STREQ(error_to_string(MetalNativeError::Success), "Success");
}

TEST(ErrorTest, ErrorToStringDeviceNotFound) {
    EXPECT_STREQ(error_to_string(MetalNativeError::DeviceNotFound), "DeviceNotFound");
}

TEST(ErrorTest, ErrorToStringAllocationFailed) {
    EXPECT_STREQ(error_to_string(MetalNativeError::AllocationFailed), "AllocationFailed");
}

TEST(ErrorTest, ErrorToStringKernelCompilationFailed) {
    EXPECT_STREQ(error_to_string(MetalNativeError::KernelCompilationFailed), "KernelCompilationFailed");
}

TEST(ErrorTest, ErrorToStringInvalidArgument) {
    EXPECT_STREQ(error_to_string(MetalNativeError::InvalidArgument), "InvalidArgument");
}

TEST(ErrorTest, ErrorToStringOutOfMemory) {
    EXPECT_STREQ(error_to_string(MetalNativeError::OutOfMemory), "OutOfMemory");
}

TEST(ErrorTest, ErrorToStringTimeoutError) {
    EXPECT_STREQ(error_to_string(MetalNativeError::TimeoutError), "TimeoutError");
}

TEST(ErrorTest, ErrorToStringInternalError) {
    EXPECT_STREQ(error_to_string(MetalNativeError::InternalError), "InternalError");
}

TEST(ErrorTest, ErrorToStringNotImplemented) {
    EXPECT_STREQ(error_to_string(MetalNativeError::NotImplemented), "NotImplemented");
}

// -- MNException tests --------------------------------------------------------

TEST(ErrorTest, ExceptionFromString) {
    MNException ex(MetalNativeError::InvalidArgument, "test message");
    EXPECT_EQ(ex.code(), MetalNativeError::InvalidArgument);
    std::string what = ex.what();
    EXPECT_NE(what.find("InvalidArgument"), std::string::npos);
    EXPECT_NE(what.find("test message"), std::string::npos);
}

TEST(ErrorTest, ExceptionFromCString) {
    MNException ex(MetalNativeError::OutOfMemory, "oom");
    EXPECT_EQ(ex.code(), MetalNativeError::OutOfMemory);
    std::string what = ex.what();
    EXPECT_NE(what.find("OutOfMemory"), std::string::npos);
    EXPECT_NE(what.find("oom"), std::string::npos);
}

TEST(ErrorTest, ExceptionIsRuntimeError) {
    MNException ex(MetalNativeError::InternalError, "test");
    // MNException derives from std::runtime_error
    const std::runtime_error& base = ex;
    EXPECT_NE(std::string(base.what()).find("test"), std::string::npos);
}

// -- MN_THROW tests -----------------------------------------------------------

TEST(ErrorTest, MNThrowThrowsException) {
    EXPECT_THROW(
        MN_THROW(MetalNativeError::InvalidArgument, "bad arg"),
        MNException
    );
}

TEST(ErrorTest, MNThrowCarriesCode) {
    try {
        MN_THROW(MetalNativeError::AllocationFailed, "alloc failed");
        FAIL() << "Expected MNException";
    } catch (const MNException& ex) {
        EXPECT_EQ(ex.code(), MetalNativeError::AllocationFailed);
    }
}

TEST(ErrorTest, MNThrowCarriesMessage) {
    try {
        MN_THROW(MetalNativeError::KernelCompilationFailed, "shader error");
        FAIL() << "Expected MNException";
    } catch (const MNException& ex) {
        std::string what = ex.what();
        EXPECT_NE(what.find("shader error"), std::string::npos);
    }
}

// -- MN_CHECK tests -----------------------------------------------------------

TEST(ErrorTest, MNCheckPassesOnTrue) {
    EXPECT_NO_THROW(
        MN_CHECK(true, MetalNativeError::InvalidArgument, "should not throw")
    );
}

TEST(ErrorTest, MNCheckThrowsOnFalse) {
    EXPECT_THROW(
        MN_CHECK(false, MetalNativeError::InvalidArgument, "check failed"),
        MNException
    );
}

TEST(ErrorTest, MNCheckCarriesCode) {
    try {
        MN_CHECK(1 == 2, MetalNativeError::TimeoutError, "timeout");
        FAIL() << "Expected MNException";
    } catch (const MNException& ex) {
        EXPECT_EQ(ex.code(), MetalNativeError::TimeoutError);
    }
}

TEST(ErrorTest, MNCheckWithExpression) {
    int x = 5;
    EXPECT_NO_THROW(
        MN_CHECK(x > 0, MetalNativeError::InvalidArgument, "x must be positive")
    );
    EXPECT_THROW(
        MN_CHECK(x < 0, MetalNativeError::InvalidArgument, "x must be negative"),
        MNException
    );
}

// -- metal_error_to_string tests ----------------------------------------------

TEST(ErrorTest, MetalErrorToStringKnownCodes) {
    EXPECT_NE(metal_error_to_string(0).find("None"), std::string::npos);
    EXPECT_NE(metal_error_to_string(1).find("Internal"), std::string::npos);
    EXPECT_NE(metal_error_to_string(2).find("Timeout"), std::string::npos);
    EXPECT_NE(metal_error_to_string(6).find("OutOfMemory"), std::string::npos);
}

TEST(ErrorTest, MetalErrorToStringUnknownCode) {
    std::string msg = metal_error_to_string(999);
    EXPECT_NE(msg.find("Unknown"), std::string::npos);
    EXPECT_NE(msg.find("999"), std::string::npos);
}
