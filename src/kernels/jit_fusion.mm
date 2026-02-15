#import <Metal/Metal.h>
#include "metal_native/kernels/jit_fusion.h"
#include "metal_native/core/device.h"
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <sstream>

namespace metal_native {

// ============================================================================
// FusedOp Helper Functions
// ============================================================================

bool is_unary_op(FusedOp op) {
    switch (op) {
        case FusedOp::Add:
        case FusedOp::Sub:
        case FusedOp::Mul:
        case FusedOp::Div:
            return false;
        default:
            return true;
    }
}

const char* fused_op_msl_expr(FusedOp op) {
    switch (op) {
        case FusedOp::Add:     return "a + b";
        case FusedOp::Sub:     return "a - b";
        case FusedOp::Mul:     return "a * b";
        case FusedOp::Div:     return "a / b";
        case FusedOp::Exp:     return "exp(x)";
        case FusedOp::Log:     return "log(x)";
        case FusedOp::Neg:     return "-x";
        case FusedOp::Abs:     return "abs(x)";
        case FusedOp::Sqrt:    return "sqrt(x)";
        case FusedOp::ReLU:    return "max(x, VEC_ZERO)";
        case FusedOp::GELU:    return "x * VEC_HALF * (VEC_ONE + tanh(VEC_GELU_CONST * (x + VEC_GELU_COEFF * x * x * x)))";
        case FusedOp::SiLU:    return "x / (VEC_ONE + exp(-x))";
        case FusedOp::Tanh:    return "tanh(x)";
        case FusedOp::Sigmoid: return "VEC_ONE / (VEC_ONE + exp(-x))";
    }
}

const char* fused_op_name(FusedOp op) {
    switch (op) {
        case FusedOp::Add:     return "add";
        case FusedOp::Sub:     return "sub";
        case FusedOp::Mul:     return "mul";
        case FusedOp::Div:     return "div";
        case FusedOp::Exp:     return "exp";
        case FusedOp::Log:     return "log";
        case FusedOp::Neg:     return "neg";
        case FusedOp::Abs:     return "abs";
        case FusedOp::Sqrt:    return "sqrt";
        case FusedOp::ReLU:    return "relu";
        case FusedOp::GELU:    return "gelu";
        case FusedOp::SiLU:    return "silu";
        case FusedOp::Tanh:    return "tanh";
        case FusedOp::Sigmoid: return "sigmoid";
    }
}

// ============================================================================
// FusionKey
// ============================================================================

bool FusionKey::operator==(const FusionKey& other) const {
    return ops == other.ops && dtype == other.dtype && num_inputs == other.num_inputs;
}

size_t FusionKeyHash::operator()(const FusionKey& k) const {
    size_t h = std::hash<uint8_t>{}(static_cast<uint8_t>(k.dtype));
    h ^= std::hash<uint32_t>{}(k.num_inputs) + 0x9e3779b9 + (h << 6) + (h >> 2);
    for (auto op : k.ops) {
        h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(op)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

// ============================================================================
// JITFusionCompiler Implementation
// ============================================================================

struct JITFusionCompiler::Impl {
    mutable std::mutex mu;
    bool enabled_ = true;

    // LRU cache: key -> pipeline
    std::unordered_map<FusionKey, id<MTLComputePipelineState>, FusionKeyHash> cache;
    std::list<FusionKey> lru_list;  // most recently used at front
    std::unordered_map<FusionKey, std::list<FusionKey>::iterator, FusionKeyHash> lru_map;

    // Pending compilations
    std::unordered_set<FusionKey, FusionKeyHash> pending;

    id<MTLDevice> device = nil;

    Impl() {
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"JITFusionCompiler: Failed to create Metal device");
        }
    }

    ~Impl() {
        std::lock_guard<std::mutex> lock(mu);
        cache.clear();
        lru_list.clear();
        lru_map.clear();
        pending.clear();
        device = nil;
    }

    void touch_lru(const FusionKey& key) {
        auto it = lru_map.find(key);
        if (it != lru_map.end()) {
            lru_list.erase(it->second);
        }
        lru_list.push_front(key);
        lru_map[key] = lru_list.begin();
    }

    void evict_if_needed() {
        while (cache.size() >= JITFusionCompiler::MAX_CACHE_SIZE && !lru_list.empty()) {
            const FusionKey& old_key = lru_list.back();
            cache.erase(old_key);
            lru_map.erase(old_key);
            lru_list.pop_back();
        }
    }
};

JITFusionCompiler::JITFusionCompiler() : impl_(std::make_unique<Impl>()) {}

JITFusionCompiler::~JITFusionCompiler() = default;

JITFusionCompiler& JITFusionCompiler::instance() {
    static JITFusionCompiler inst;
    return inst;
}

bool JITFusionCompiler::enabled() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->enabled_;
}

void JITFusionCompiler::set_enabled(bool enable) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->enabled_ = enable;
}

bool JITFusionCompiler::has_cached(const FusionKey& key) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->cache.find(key) != impl_->cache.end();
}

#ifdef __OBJC__
id<MTLComputePipelineState> JITFusionCompiler::get_pipeline(const FusionKey& key) {
#else
void* JITFusionCompiler::get_pipeline(const FusionKey& key) {
#endif
    std::unique_lock<std::mutex> lock(impl_->mu);

    if (!impl_->enabled_) {
        return nil;
    }

    if (!impl_->device) {
        return nil;
    }

    // Check cache
    auto cache_it = impl_->cache.find(key);
    if (cache_it != impl_->cache.end()) {
        impl_->touch_lru(key);
        return cache_it->second;
    }

    // Check if already compiling
    if (impl_->pending.find(key) != impl_->pending.end()) {
        return nil;  // Still compiling, caller should fall back
    }

    // Limit concurrent compilations to avoid resource exhaustion
    static constexpr size_t MAX_PENDING = 16;
    if (impl_->pending.size() >= MAX_PENDING) {
        return nil;  // Too many in-flight compilations
    }

    // Mark as pending
    impl_->pending.insert(key);

    // Generate MSL source
    std::string msl_source = generate_msl(key);

    // Copy necessary data for async compilation
    FusionKey key_copy = key;
    id<MTLDevice> device = impl_->device;
    auto impl_ptr = impl_.get();

    lock.unlock();

    // Compile asynchronously
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSError* error = nil;
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, iOS 18.0, *)) {
                opts.mathMode = MTLMathModeRelaxed;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }

            id<MTLLibrary> lib = [device newLibraryWithSource:@(msl_source.c_str())
                                                      options:opts
                                                        error:&error];
            if (error) {
                NSLog(@"JIT fusion compile error: %@", error);
                std::lock_guard<std::mutex> lock(impl_ptr->mu);
                impl_ptr->pending.erase(key_copy);
                return;
            }

            if (!lib) {
                NSLog(@"JIT fusion: Failed to create library");
                std::lock_guard<std::mutex> lock(impl_ptr->mu);
                impl_ptr->pending.erase(key_copy);
                return;
            }

            size_t hash = FusionKeyHash{}(key_copy);
            std::string kernel_name = "fused_ew_" + std::to_string(hash);

            id<MTLFunction> func = [lib newFunctionWithName:@(kernel_name.c_str())];
            if (!func) {
                NSLog(@"JIT fusion: Failed to find kernel function %s", kernel_name.c_str());
                std::lock_guard<std::mutex> lock(impl_ptr->mu);
                impl_ptr->pending.erase(key_copy);
                return;
            }

            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
            if (error || !pipeline) {
                NSLog(@"JIT fusion: Failed to create pipeline state: %@", error);
                std::lock_guard<std::mutex> lock(impl_ptr->mu);
                impl_ptr->pending.erase(key_copy);
                return;
            }

            // Cache the pipeline
            {
                std::lock_guard<std::mutex> lock(impl_ptr->mu);
                impl_ptr->evict_if_needed();
                impl_ptr->cache[key_copy] = pipeline;
                impl_ptr->touch_lru(key_copy);
                impl_ptr->pending.erase(key_copy);
            }
        }
    });

    return nil;  // Return nullptr on first call, cached pipeline available on subsequent calls
}

std::string JITFusionCompiler::generate_msl(const FusionKey& key) const {
    if (key.ops.empty() || key.ops.size() > MAX_CHAIN_LENGTH) {
        return "";
    }

    // Validate num_inputs matches the op sequence requirements:
    // First input is always consumed, each binary op consumes one additional input.
    uint32_t required_inputs = 1;
    for (auto op : key.ops) {
        if (!is_unary_op(op)) {
            required_inputs++;
        }
    }
    if (key.num_inputs != required_inputs) {
        return "";
    }

    const char* vec_type = (key.dtype == FusedDType::Float32) ? "float4" : "half4";
    const char* zero_literal = (key.dtype == FusedDType::Float32) ? "0.0f" : "0.0h";
    const char* one_literal = (key.dtype == FusedDType::Float32) ? "1.0f" : "1.0h";
    const char* half_literal = (key.dtype == FusedDType::Float32) ? "0.5f" : "0.5h";

    // Create unique kernel name from hash
    size_t hash = FusionKeyHash{}(key);
    std::string kernel_name = "fused_ew_" + std::to_string(hash);

    std::ostringstream src;
    src << "#include <metal_stdlib>\n";
    src << "using namespace metal;\n\n";

    // Define vector constants as macros
    src << "#define VEC_ZERO " << vec_type << "(" << zero_literal << ")\n";
    src << "#define VEC_ONE " << vec_type << "(" << one_literal << ")\n";
    src << "#define VEC_HALF " << vec_type << "(" << half_literal << ")\n";
    src << "#define VEC_GELU_CONST " << vec_type << "(0.7978845608" << (key.dtype == FusedDType::Float32 ? "f" : "h") << ")\n";
    src << "#define VEC_GELU_COEFF " << vec_type << "(0.044715" << (key.dtype == FusedDType::Float32 ? "f" : "h") << ")\n\n";

    // Build kernel signature
    src << "kernel void " << kernel_name << "(\n";

    uint32_t buffer_idx = 0;
    // Add input buffers
    for (uint32_t i = 0; i < key.num_inputs; i++) {
        src << "    device const " << vec_type << "* input" << i
            << " [[buffer(" << buffer_idx++ << ")]],\n";
    }
    // Output buffer
    src << "    device " << vec_type << "* output [[buffer(" << buffer_idx++ << ")]],\n";
    // num_vec4s
    src << "    constant uint& num_vec4s [[buffer(" << buffer_idx++ << ")]],\n";
    src << "    uint id [[thread_position_in_grid]])\n";
    src << "{\n";
    src << "    if (id >= num_vec4s) return;\n\n";

    // Load first input
    src << "    " << vec_type << " v = input0[id];\n";

    // Generate operations
    // Helper lambda: emit MSL expression for each op directly (avoids
    // naive char replacement that corrupts function names like max/exp).
    uint32_t next_input = 1;
    for (size_t i = 0; i < key.ops.size(); i++) {
        FusedOp op = key.ops[i];

        if (is_unary_op(op)) {
            switch (op) {
                case FusedOp::Exp:     src << "    v = exp(v);\n"; break;
                case FusedOp::Log:     src << "    v = log(v);\n"; break;
                case FusedOp::Neg:     src << "    v = -v;\n"; break;
                case FusedOp::Abs:     src << "    v = abs(v);\n"; break;
                case FusedOp::Sqrt:    src << "    v = sqrt(v);\n"; break;
                case FusedOp::ReLU:    src << "    v = max(v, VEC_ZERO);\n"; break;
                case FusedOp::GELU:    src << "    v = v * VEC_HALF * (VEC_ONE + tanh(VEC_GELU_CONST * (v + VEC_GELU_COEFF * v * v * v)));\n"; break;
                case FusedOp::SiLU:    src << "    v = v / (VEC_ONE + exp(-v));\n"; break;
                case FusedOp::Tanh:    src << "    v = tanh(v);\n"; break;
                case FusedOp::Sigmoid: src << "    v = VEC_ONE / (VEC_ONE + exp(-v));\n"; break;
                default: break;
            }
        } else {
            // Binary: read next input
            if (next_input >= key.num_inputs) {
                // Invalid: not enough inputs for binary op
                return "";
            }
            src << "    {\n";
            src << "        " << vec_type << " b = input" << next_input++ << "[id];\n";
            switch (op) {
                case FusedOp::Add: src << "        v = v + b;\n"; break;
                case FusedOp::Sub: src << "        v = v - b;\n"; break;
                case FusedOp::Mul: src << "        v = v * b;\n"; break;
                case FusedOp::Div: src << "        v = v / b;\n"; break;
                default: break;
            }
            src << "    }\n";
        }
    }

    src << "    output[id] = v;\n";
    src << "}\n";

    return src.str();
}

void JITFusionCompiler::invalidate_all() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->cache.clear();
    impl_->lru_list.clear();
    impl_->lru_map.clear();
    // Don't clear pending - let in-flight compilations finish
}

size_t JITFusionCompiler::cache_size() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->cache.size();
}

size_t JITFusionCompiler::pending_compilations() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->pending.size();
}

} // namespace metal_native
