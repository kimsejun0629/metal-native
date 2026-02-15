<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-12 | Updated: 2026-02-12 -->
# graph
## Purpose
Implementation files for MPSGraph infrastructure. Implements graph builder, two-level cache, lazy evaluation, fusion patterns, serialization, and shape bucketing.
## Key Files
| File | Description |
|------|-------------|
| CMakeLists.txt | Build configuration for graph module |
| graph_builder.mm | GraphBuilder implementation (MPSGraph fluent API wrapper) |
| graph_cache.mm | GraphCache implementation (L1 LRU + L2 disk cache with eviction timer) |
| lazy_graph.mm | LazyGraph implementation (DAG recording, topological eval, optimization passes) |
| fusion_patterns.mm | FusionRegistry implementation (pattern matching for fusible op sequences) |
| graph_serializer.mm | GraphSerializer implementation (serialize to .mpsgraphpackage with metadata) |
| shape_bucketing.mm | ShapeBucketer implementation (round to predefined power-of-2 buckets) |
## For AI Agents
### Working In This Directory
- **graph_builder.mm**: Wraps MPSGraph construction with fluent API, tracks placeholders and outputs
- **graph_cache.mm**: L1 uses std::unordered_map + std::list for LRU; L2 writes to ~/.cache/metal_native/
- **lazy_graph.mm**: Builds DAG of LazyNode structs, runs topological sort on eval(), applies fusion
- **fusion_patterns.mm**: Pattern matching with greedy left-to-right scan, longest match wins
- **graph_serializer.mm**: Uses MPSGraphExecutable's serialize API, stores version info in JSON
- **shape_bucketing.mm**: Buckets: 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384
### Common Patterns
```objc
// Graph builder
MPSGraph* graph = [[MPSGraph alloc] init];
// ... add operations ...
MPSGraphExecutable* exec = [graph compileWithDevice:device
                                              feeds:feedsDict
                                   targetOperations:outputs
                                            options:nil];

// Cache key construction
uint64_t hash = compute_topology_hash(graph);
std::vector<size_t> shapes = flatten_shapes(inputs);
GraphCacheKey key{hash, shapes, dtype};

// Lazy graph DAG
LazyNode node;
node.op_type = OpType::MatMul;
node.inputs = {lhs_id, rhs_id};
node.output_shape = infer_shape(lhs_shape, rhs_shape);
dag_.insert({node_id, node});
```
## Dependencies
### Internal
- ../include/metal_native/graph/ (public headers)
- core/ (MNDevice, MNShape, MNDType, MNTensor)
### External
- MetalPerformanceShadersGraph framework (MPSGraph, MPSGraphExecutable)
- Foundation framework (file I/O for cache)
<!-- MANUAL: -->
