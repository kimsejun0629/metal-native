#pragma once

/// @file dlpack_header.h
/// @brief DLPack tensor exchange protocol header (v0.8 compatible).
///
/// DLPack is a standard for zero-copy tensor sharing between frameworks.
/// This header defines the core types from the DLPack specification.

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Device types
// ---------------------------------------------------------------------------

/// Device type codes as defined by DLPack specification.
typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
} DLDeviceType;

/// Device context for a DLTensor.
typedef struct {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Data type codes as defined by DLPack specification.
typedef enum {
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
    kDLBfloat = 4U,
} DLDataTypeCode;

/// Data type descriptor.
typedef struct {
    uint8_t code;   ///< DLDataTypeCode
    uint8_t bits;   ///< Number of bits (8, 16, 32, 64, etc.)
    uint16_t lanes; ///< Number of lanes (1 for scalar, >1 for vector types)
} DLDataType;

// ---------------------------------------------------------------------------
// Tensor descriptor
// ---------------------------------------------------------------------------

/// Plain DLTensor structure without ownership semantics.
typedef struct {
    void* data;             ///< Pointer to the tensor data
    DLDevice device;        ///< Device where the tensor resides
    int32_t ndim;           ///< Number of dimensions
    DLDataType dtype;       ///< Data type descriptor
    int64_t* shape;         ///< Array of dimension sizes (length = ndim)
    int64_t* strides;       ///< Array of strides in elements (length = ndim, can be NULL)
    uint64_t byte_offset;   ///< Byte offset from data pointer to first element
} DLTensor;

// ---------------------------------------------------------------------------
// Managed tensor (with ownership)
// ---------------------------------------------------------------------------

/// Managed DLTensor with reference counting and deletion callback.
///
/// The consumer of a DLManagedTensor should call the deleter function
/// exactly once when it is done with the tensor.
typedef struct DLManagedTensor {
    DLTensor dl_tensor;     ///< The actual tensor descriptor
    void* manager_ctx;      ///< Context pointer for the deleter (opaque)
    void (*deleter)(struct DLManagedTensor* self);  ///< Deletion callback
} DLManagedTensor;

#ifdef __cplusplus
}  // extern "C"
#endif
