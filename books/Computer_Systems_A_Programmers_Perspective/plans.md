-----------------------------------------------------------------------------
John Doe's .plan (nanodijkstra engineering log)                2023.08.25
-----------------------------------------------------------------------------

Recent work:
- Implemented hot-reloading system for game assets using inotify
- Prototyped entity component system with 16-byte cache-line alignment
- Rewrote matrix stack using SSE intrinsics (3.2x speedup on profile_batch_transform)
- Fixed memory stomp in texture atlas allocator (off-by-one in bin packing)

Current focus:
! Reduce frame spikes in physics subsystem
! Investigate context-switch costs in job system
! GLES3 backend for Android port

Technical deep:
The new SIMD math library uses union type-punning for vector lanes:
typedef union {
    __m128 v;
    struct { float x, y, z, w; };
} vec4;

This generates better code than _mm_set_ps() while keeping syntax clean.
Tradeoff: Breans strict aliasing (-fno-strict-aliasing required)

Optimization wins:
- Switched to 4KB stack-allocated scratch buffers for small string ops
- Reduced L2 cache misses 18% by reorganizing particle data (SOA -> AOSOA)
- Discovered 7-cycle stall in texture sampler - fixed with __builtin_prefetch

Anti-patterns:
✗ Avoided "clever" macro system for ECS - maintainability matters
✗ Rejected virtual functions for render backend - vtable costs too high
✗ Removed alloca() usage - security audit flagged stack exhaustion risk

Next:
> Finish WASM port with emscripten pthread support
> Prototype mesh shader pipeline
> Write SIMD coverage tests using μbench

Carmackism of the day:
"A complex system that works is invariably found to have evolved from a simple system
that worked. The inverse proposition also appears to be true."

-----------------------------------------------------------------------------
Compiled with clang-16 -Wall -Wextra -Werror -O3 -march=native
-----------------------------------------------------------------------------
