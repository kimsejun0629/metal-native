<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# graph
## Purpose
MPSGraph computation graph infrastructure. Provides fluent graph builder, two-level cache (in-memory + disk), lazy evaluation framework, fusion pattern matching, graph serialization, and shape bucketing for reduced recompilation.
## Key Files
| File | Description |
|------|-------------|
| graph_builder.h | GraphBuilder: fluent API for constructing MPSGraph with automatic dtype propagation |
| graph_cache.h | GraphCache: two-level LRU cache (L1 in-memory, L2 disk-backed .mpsgraphpackage) |
| lazy_graph.h | LazyGraph: deferred computation DAG with optimization passes (fusion, DCE) |
| fusion_patterns.h | FusionRegistry: pattern matching for operation fusion (Conv+BN+ReLU, Linear+GELU, etc.) |
| graph_serializer.h | GraphSerializer: serialize/deserialize MPSGraphExecutable with version validation |
| shape_bucketing.h | ShapeBucketer: round shapes to predefined buckets to reduce graph recompilation |
## For AI Agents
### Working In This Directory
- **GraphBuilder** provides fluent API: `builder.add_placeholder(...).add_matmul(...).add_relu(...).build()`
- **GraphCache** uses topology hash + shape tuple + dtype as key; L2 cache survives process restarts
- **LazyGraph** records operations without execution until `eval()` or `eval_all()` is called
- **FusionRegistry** identifies fusible patterns to reduce kernel launches and memory traffic
- **ShapeBucketer** rounds dynamic shapes (e.g., batch=37 -> 48) to allow graph reuse
### Common Patterns
```cpp
// Build and compile graph
GraphBuilder builder(device);
auto x = builder.add_placeholder("x", {1, 256}, MNDType::Float32);
auto w = builder.add_placeholder("w", {256, 512}, MNDType::Float32);
auto y = builder.add_matmul(x, w);
auto z = builder.add_relu(y);
builder.mark_output("z");
GraphExecutable executable = builder.build();

// Cache lookup
GraphCacheKey key{topology_hash, shape_tuple, dtype};
MPSGraphExecutable* cached = graph_cache.lookup(key);
if (!cached) {
    cached = /* compile */;
    graph_cache.insert(key, cached);
}

// Lazy evaluation
LazyGraph& lazy = LazyGraph::instance();
LazyNodeId node = lazy.record_binary(OpType::MatMul, lhs_id, rhs_id, shape, dtype);
// ... record more ops ...
lazy.optimize_fusions();  // identify fusible patterns
MNTensor& result = lazy.eval(node);  // materialize
```
## Dependencies
### Internal
- core/ (MNDevice, MNShape, MNDType, MNTensor)
### External
- MetalPerformanceShadersGraph framework (MPSGraph, MPSGraphExecutable)
<!-- MANUAL: -->
