# MetalNative Optimization Plan

> Apple M4 Max | 36GB | PyTorch 2.8.0 기준 분석
> 생성일: 2026-02-11

---

## Executive Summary

4개 영역(Metal Shaders, Memory/Dispatch, Graph/Fusion, 최신 기술 연구)에 대한 심층 분석 결과,
MetalNative에는 **정합성 버그 4건**, **핵심 성능 병목 7건**, **아키텍처 개선 기회 12건**이 식별되었습니다.

가장 큰 성능 기회는 다음 3가지입니다:
1. **Attention 커널에 `simdgroup_matrix_multiply` 적용** → 10-30x 속도 향상
2. **MPSGraph 캐시를 Op 구현에 연결** → 반복 추론 시 2-3x 향상
3. **Normalization/Reduction 커널 병렬화** → 32-256x 속도 향상

---

## Phase 0: 긴급 버그 수정 (P0 - Critical)

### BUG-1: Softmax SIMD Reduction 정합성 버그
- **파일**: `shaders/softmax_kernel.metal:37-42, 74-80`
- **문제**: 각 스레드가 독립적으로 자기 row의 max/sum을 계산하지만, `simd_max_reduce`/`simd_sum_reduce`가 **서로 다른 row**의 결과끼리 reduce. Lane 0만 write하므로 31/32 출력값이 유실됨.
- **수정**: 그리드를 1D로 변경하여 threadgroup이 하나의 row를 협력 처리하거나, SIMD reduction을 제거하고 각 스레드가 자기 결과를 직접 write.
- **영향**: 정합성 수정 (현재 결과가 틀림)

### BUG-2: Attention head_dim > 32 버퍼 오버플로우
- **파일**: `shaders/attention_kernel.metal:86`
- **문제**: `float acc[32]`로 고정. head_dim=64/128(현대 모든 모델)에서 out-of-bounds write.
- **수정**: 동적 크기 또는 template specialization per head_dim.
- **영향**: Silent data corruption 방지

### BUG-3: Attention threadgroup 메모리 32KB 초과
- **파일**: `src/ops/attention.mm:133-134`
- **문제**: head_dim=128 FP32에서 `3*32*128*4 + 32*32*4 = 53KB` → Apple Silicon 32KB 한도 초과.
- **수정**: TILE_SIZE를 16으로 축소하거나 head_dim을 타일링.
- **영향**: GPU fault 방지

### BUG-4: commit_and_continue 활성도/기아(liveness/starvation) 위험
- **파일**: `src/dispatch/command_pipeline.mm:78-100`
- **문제**: `impl_->mu` mutex를 잡은 상태에서 `backpressure.acquire()` 블로킹 호출. Backpressure `release()`는 자체 mutex를 사용하므로 classic deadlock은 아니나, `current_buffer()` 등 `impl_->mu`가 필요한 다른 스레드가 블로킹되어 convoy/starvation 발생.
- **수정**: backpressure acquire를 mutex 획득 **전**으로 이동.
- **영향**: 스레드 기아 및 convoy 방지

---

## Phase 1: 핵심 성능 최적화 (1-2주)

### OPT-1: Attention 커널 - simdgroup_matrix_multiply 적용
- **현재**: 스칼라 dot-product 루프 (`attention_kernel.metal:114-118`)
- **목표**: Apple Silicon AMX 코프로세서 활용
- **방법**:
  - Q@K^T와 attn_weights@V에 `simdgroup_matrix_multiply_accumulate` (8x8 타일) 사용
  - TILE_SIZE를 16으로 줄여 32KB 내 유지
  - head_dim을 8의 배수로 타일링
- **예상 효과**: Attention 연산 **10-30x 속도 향상**
- **참고**: [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) 구현 참조

### OPT-2: Reduction 커널 완전 재작성
- **현재**: 단일 스레드가 전체 reduction 차원을 순차 처리 (`reduction_kernel.metal:33-36`)
- **목표**: 2단계 병렬 reduction (SIMD-group → threadgroup)
- **방법**:
  ```
  1. 각 스레드가 stride만큼 부분합 계산
  2. simd_sum으로 SIMD group 내 reduce
  3. threadgroup memory에 SIMD group 결과 저장
  4. threadgroup barrier 후 최종 reduce
  ```
- **예상 효과**: vocab_size=32K reduction에서 **50-256x 속도 향상**

### OPT-3: Normalization 커널 병렬화
- **현재**:
  - LayerNorm: 단일 스레드 Welford loop (`normalization_kernel.metal:53-55`)
  - BatchNorm: threadgroup (1,1,1) 디스패치 (`normalization.mm:220`)
  - GroupNorm: threadgroup (1,1,1) 디스패치 (`normalization.mm:328`)
- **목표**: SIMD-cooperative 통계 계산
- **방법**:
  - 32 스레드가 norm_size를 stride로 분할 → 부분 통계 → `simd_sum`으로 결합
  - BatchNorm: batch*spatial 차원에 걸쳐 병렬화
  - GroupNorm: 동일 패턴 적용
- **예상 효과**: LayerNorm **32x**, BatchNorm/GroupNorm **100-1000x 속도 향상**

### OPT-4: MPSGraph 캐시 연결
- **현재**: 매 호출마다 새 MPSGraph 생성 (`matmul.mm:67`, `conv.mm:68`)
- **기존 인프라**: `GraphCache` (L1 LRU + L2 disk) 완전 구현되어 있으나 **미연결**
- **방법**:
  - Op 구현에서 (topology, shape, dtype) 해시로 캐시 lookup
  - 캐시 미스 시에만 build + compile
  - `compilationDescriptor`에 최적화 레벨 설정
- **예상 효과**: 반복 추론 시 **2-3x 속도 향상** (그래프 구축 오버헤드 제거)

### OPT-5: Per-Op 동기화 제거
- **현재**: 모든 Op에 `waitUntilCompleted` 호출 (21곳)
- **문제**: Transformer block 15개 Op × ~0.1ms = ~1.5ms 순수 동기화 오버헤드
- **방법**: Command buffer 배칭 - 논리적 계산 단위(transformer block) 끝에서만 동기화
- **예상 효과**: End-to-end **15-30% 속도 향상**

---

## Phase 2: 메모리 & 디스패치 최적화 (2-4주)

### OPT-6: 소규모 텐서 CPU 패스트패스
- **근거**: 벤치마크에서 128x128 MatMul이 MPS에서 CPU 대비 0.03x (33배 느림)
- **원인**: MPSGraph 구축 + 커널 디스패치 + GPU 동기화 오버헤드가 연산 자체보다 큼
- **방법**: 임계값(예: 총 요소 < 65536) 이하에서 CPU 실행 또는 경량 Metal 커널로 분기
- **예상 효과**: 소규모 연산 **10-30x 속도 향상**, end-to-end 혼합 워크로드 개선

### OPT-7: Elementwise 커널 벡터화 (float4/half4)
- **현재**: 스레드당 1개 요소 처리, `float4`/`half4` 미사용
- **방법**: `device const float4*` 캐스팅으로 4요소씩 처리 + tail handling
- **예상 효과**: **2-4x 속도 향상**
- **적용 대상**: 26개 elementwise 커널 + flat copy

### OPT-7: Softmax 단일 패스 온라인 알고리즘
- **현재**: 3-pass (find_max → exp_sum → normalize), 3개 별도 커널 디스패치
- **목표**: attention 커널 내부에 이미 있는 online softmax 패턴 재활용
- **예상 효과**: **2-3x 속도 향상** (대역폭 절반 + 임시 버퍼 제거)

### OPT-8: 메모리 할당기 중간 크기 클래스
- **현재**: 순수 power-of-2 버킷팅 → 최대 49.99% 내부 단편화
- **방법**: 1.5x 중간 클래스 추가 (16KB, 24KB, 32KB, 48KB, 64KB, 96KB...)
- **예상 효과**: 피크 GPU 메모리 **15-25% 절감**

### OPT-9: MTLHazardTrackingModeUntracked 설정
- **파일**: `src/memory/heap_manager.mm:141-142`
- **방법**: Placement heap에 `desc.hazardTrackingMode = MTLHazardTrackingModeUntracked` (1줄 변경)
- **전제**: EventManager로 이미 동기화 관리 중
- **예상 효과**: GPU 처리량 **3-8% 향상**

### OPT-10: Worker Thread drain 최적화
- **현재**: 매 task 완료 후 mutex 재획득하여 queue.empty() 확인
- **방법**: `std::atomic<bool> drain_requested` 플래그 추가
- **예상 효과**: 소규모 커널 집약 워크로드에서 **5-15% 처리량 향상**

### OPT-11: 중간 텐서 StorageMode::Private 기본값
- **현재**: 모든 할당이 `Shared` 모드 (CPU 페이지 테이블 매핑 유지)
- **방법**: GPU 전용 중간 텐서를 `Private` 모드로 변경
- **예상 효과**: TLB 압력 감소로 **2-10% 처리량 향상**

---

## Phase 3: 그래프 최적화 & 커널 퓨전 (1-2개월)

### OPT-12: Fusion 패턴 연결 및 확장
- **현재**: 7개 패턴 등록됨 (`fusion_patterns.mm:132-197`) 그러나 **실행 경로 미연결**
- **추가할 패턴**:
  | 패턴 | 설명 | 예상 속도 향상 |
  |------|------|--------------|
  | MatMul + GELU | FFN 핵심 패턴 | 1.25x |
  | QKV Projection Fusion | 3개 matmul → 1개 큰 matmul | 1.3x |
  | LayerNorm/RMSNorm + Linear | Norm 출력 → matmul 직접 연결 | 1.2x |
  | Residual Add + LayerNorm | Transformer 핵심 패턴 | 1.15x |
  | SiLU + Gate (Llama/Mistral) | `SiLU(x_gate) * x_up` | 2x |
  | Attention + Dropout | 학습용 | 1.1x |

### OPT-13: Shape Bucketing 완성
- **현재**: `pad_tensor()`, `slice_tensor()` → NotImplemented 예외
- **방법**: MPSGraph `padTensorWithTensor:paddingMode:` 활용
- **차원별 전략**:
  - Batch: {1, 2, 4, 8, 16, 32, 64}
  - Sequence: power-of-2 only (타일링 정렬)
  - Hidden: 이미 정렬된 경우 버킷팅 생략
- **예상 효과**: 그래프 컴파일 **20-50% 감소**

### OPT-14: L2 디스크 캐시 키 충돌 수정
- **현재**: `topology_hash`만으로 디스크 경로 생성 → 다른 shape의 그래프가 서로 덮어씀
- **수정**: `shape_tuple` 해시 + `dtype`을 파일명에 포함

### OPT-15: Lazy Evaluation 아키텍처 (MLX 영감)
- **MLX 핵심 장점**: 계산 그래프를 실행과 분리 → 전역 최적화 패스 적용
- **MetalNative 적용**:
  1. Operation을 즉시 실행하지 않고 그래프에 기록
  2. DFS 의존성 분석 → BFS 실행 계획
  3. 자동 퓨전 탐지 및 재작성
  4. 상수 접기, 데드 코드 제거
- **예상 효과**: End-to-end **1.5-3x 향상** (MLX는 PyTorch MPS 대비 25-30x)

---

## Phase 4: 차세대 하드웨어 대응 (3-6개월)

### OPT-16: Metal 4 마이그레이션
- `MTLTensor` 네이티브 텐서 타입 채택
- `MTL4ArgumentTable`로 효율적 리소스 바인딩
- `MTL4MachineLearningCommandEncoder`로 ML 전용 인코더
- Residency sets로 커널/버퍼 사전 로딩
- Shader precompilation via MTL4Compiler

### OPT-17: BFloat16 지원
- M4 Max 하드웨어 지원 확인
- FP32 동적 범위 + FP16 처리량 → 학습 안정성
- `AMPController`에 BF16 정책 추가

### OPT-18: M5 Neural Accelerator 대응 (Metal 4 TensorOps 경유)
- Metal FlashAttention v2.5: **4.6x 성능 향상** 보고
- 전용 matmul, attention, segmented matmul 연산
- M5 칩 대상 최적화 경로

### OPT-19: `metal_native.fast` 모듈 (MLX 영감)
- 고도 최적화된 퓨전 커널 라이브러리:
  - `fast.rms_norm()` - 퓨전 RMSNorm
  - `fast.layer_norm()` - 퓨전 LayerNorm
  - `fast.rope()` - Rotary Position Embedding
  - `fast.scaled_dot_product_attention()` - Flash Attention
  - `fast.swiglu()` - SiLU + Gate 퓨전
- MLX에서 GeLU 10-50x 속도 향상 사례

---

## 우선순위 종합 매트릭스

| 순위 | 항목 | 노력 | 예상 효과 | 위험 |
|------|------|------|----------|------|
| **P0** | BUG-1: Softmax 정합성 | 수시간 | 정합성 수정 | 낮음 |
| **P0** | BUG-2: Attention head_dim | 1일 | 실제 모델 지원 | 낮음 |
| **P0** | BUG-3: Attention 메모리 초과 | 1일 | GPU fault 방지 | 낮음 |
| **P0** | BUG-4: 스레드 기아 수정 | 수시간 | 정합성 수정 | 낮음 |
| **P1** | OPT-1: simdgroup_matrix Attention | 수일 | **10-30x** | 높음 |
| **P1** | OPT-2: Reduction 병렬화 | 수일 | **50-256x** | 중간 |
| **P1** | OPT-3: Normalization 병렬화 | 수일 | **32-1000x** | 중간 |
| **P1** | OPT-4: MPSGraph 캐시 연결 | 수일 | **2-3x** (반복 추론) | 낮음 |
| **P1** | OPT-5: Per-Op 동기화 제거 | 수일 | **15-30%** | 중간 |
| **P1** | OPT-14: L2 캐시 키 충돌 수정 | 수시간 | 정합성 (OPT-4 전제조건) | 낮음 |
| **P2** | OPT-6: 소규모 텐서 CPU 패스트패스 | 수일 | **10-30x** (소규모) | 낮음 |
| **P2** | OPT-7: Elementwise 벡터화 | 1일 | **2-4x** | 낮음 |
| **P2** | OPT-8: Online Softmax | 1일 | **2-3x** | 낮음 |
| **P2** | OPT-9: 할당기 중간 크기 | 수일 | 메모리 15-25% | 낮음 |
| **P2** | OPT-10: Untracked hazard | 수시간 | **3-8%** | 낮음 |
| **P2** | OPT-11: Worker drain 최적화 | 수시간 | **5-15%** | 낮음 |
| **P2** | OPT-12: Private 모드 기본값 | 수일 | **2-10%** | 중간 |
| **P3** | OPT-13: Fusion 패턴 연결 | 1-2주 | **1.15-2x** (패턴별) | 중간 |
| **P3** | OPT-14: Shape Bucketing 완성 | 1주 | 컴파일 20-50% 감소 | 중간 |
| **P3** | OPT-15: Lazy Evaluation | 2-4주 | **1.5-3x** | 높음 |
| **P4** | OPT-16: Metal 4 마이그레이션 | 1-2개월 | 차세대 지원 | 높음 |
| **P4** | OPT-17: BFloat16 | 2-3주 | 학습 안정성 | 중간 |
| **P4** | OPT-18: M5 Neural Accelerator | 1개월 | **4.6x** (attention) | 높음 |
| **P4** | OPT-19: fast 모듈 | 1-2개월 | **10-50x** (fused ops) | 중간 |

---

## 핵심 근본 원인 (Root Cause)

### 1. "One thread per element" CUDA 사고방식
현재 커널들은 Apple Silicon의 협력적 threadgroup/SIMD-group 프리미티브를 활용하지 않음.
`simdgroup_matrix_multiply_accumulate`, `simd_shuffle`, vectorized float4/half4 로드가 전혀 사용되지 않음.

### 2. 분리된 최적화 파이프라인
GraphCache, ShapeBucketing, FusionRegistry, MixedPrecision이 모두 독립 구현되어 있으나
실제 실행 경로에 연결되지 않음. Op들이 GraphBuilder를 완전히 우회.

### 3. MLX 대비 아키텍처 격차
MLX의 핵심 장점(lazy evaluation, 자동 fusion, UMA 최적화)이 MetalNative에 부재.
MLX는 동일 하드웨어에서 PyTorch MPS 대비 25-30x 빠른 LLM 추론 달성.

---

## 벤치마크 기반 검증 계획

각 최적화 적용 후 다음 벤치마크로 효과 측정:

```bash
# 전체 벤치마크
python benchmarks/benchmark_comprehensive.py --output benchmark_results.json

# 시각화
python benchmarks/visualize_benchmark.py --input benchmark_results.json --output-dir benchmark_charts
```

**목표 성능** (Phase 1-2 완료 후):
- MatMul 4096x4096: 13.86ms → ~5ms (AMX 활용)
- Attention B1_H32_S2048_D128: 11.5ms → ~1ms (simdgroup_matrix)
- LayerNorm Llama-7B: 0.26ms → ~0.05ms (SIMD 병렬화)
- Transformer Block Llama-7B: 22.9ms → ~5ms (종합 최적화)
