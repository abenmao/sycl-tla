# Add Xe4 Block Scaled (MXFP8/6/4, NVFP8/6/4+) Support

Xe4 Async-MMA instruction supports block scaled format for input matrices. Each matrix can be block-scaled or an arbitraty floating point type.

PISA instruction

```
async_gmma.shape.type<.ascale><.bscale><.am><.bk><.d-tm><.a-tm><.b-tm><.a_reuse> ctrl, d-desc, a-desc, b-desc, c-desc <, a-scale-desc> <, b-scale-desc> <, [d-barrier]> <, [a-barrier] <, a-wg-mask>> <, [b-barrier] <, b-wg-mask>>
```
```
async_gmma.shape.type<.ascale><.bscale><.am><.bk><.d-tm><.a-tm><.b-tm><.a_reuse> ctrl, d-desc, a-desc, b-desc <, a-scale-desc> <, b-scale-desc> <, [d-barrier]> <, [a-barrier] <, a-wg-mask>> <, [b-barrier] <, b-wg-mask>>
```

For more detailed information see [`async_gmma` PISA documentation](https://curly-invention-299nr7q.pages.github.io/future/html/systolic_2.0/asynchronous_gmma.html#async-gmma-mx-data).

## Data Type Description

Block scaled format is numeric representation where a group (block) of values shares a common scale factor, in addition to their own exponent. They sit conceptually between floating point and integer quantization, and are widely used in AI training/inference, signal processing, and hardware accelerators to balance dynamic range, precision, and efficiency.

GPU docs refer to the scale factor also as MX metadata; both terms are used interchangeably in the following description.

### Representation in SYCL-TLA

Original code base has supported block scaled format, which uses wrapper type to augment scale factor type on top of narrow precision type. For example, in addition to fp8, mxfp8 requires extra e8m0 scale for each block of data which can be captured by template:

```
template <class F8Type>
struct mx_float8_t {
  using ScaleFactorType = cutlass::float_ue8m0_t;
  using DataType = F8Type;
};
```

In order to support nvfp4+, we shall implement `float_ue5m3_t` following the pattern of `float_ue8m0_t`. For more detailed block scaled datat types see []

### Mixed Precision Support

Different floating-point data types can be mixed in Async-MMA to support a wide range of use cases. For example, Async-MMA instruction supports combinations such as MXFP8 with MXFP4 or FP16 with NVFP4. Supported combinations are constrained only by hardware capabilities, not by the software implementation. Exhaustive enumeration is not required in the current implementation but left possibility for adding them in the future.

## Scale Factor Layout
### Layout Deduction

Each block scaled matrix has a scale factor (SF) matrix attached to it. Both of their layouts for local and global accesses are correlated. In order to infer SF matrix layout from its master matrix, we need to implement ``Xe4BlockScaledConfig`` type in ``xe4_blockscaled_layout.hpp`` which capture the layout relationship that follows Xe4's hardware specs.

### SF Layout Description

Xe4 scale factor matrix is MN-Major nature format. Different than B200 which use `((_32,_4),(_16,_4)):((_16,_4),(_0,_1))` block (128(mn) x 4(k) tiled in 32-bit, 32-row, 4-column, single 128-bit full warp load friendly).

```
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {

  using Blk_MN    = _128;
  using Blk_SF    =   _4; 

  using SfKMajorAtom  = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>, 
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_32,_4>>, 
                               Stride<Stride<            _0, _1>, Stride<_16,_4>>>;
  using SfAtom    = cute::conditional_t<major == UMMA::Major::K, SfKMajorAtom, SfMNMajorAtom>;
};
```

For basic scaling factor block, two modes were used to describe it: column block and row block. Xe4 exposes normal MN-major matrix layout hence row block is `_1` and column block is `(SFVecSize, _1)`.

```
template<int SFVecSize, AMMA::Major major = AMMA::Major::MN>
struct Xe4BlockScaledBasicChunk {

  using Blk_MN    = _1;
  using Blk_SF    = _1;

  using SfMNMajorAtom = Layout< Shape< _1, Int<SFVecSize>>,
                                Stride<_1,             _0>>;
  using SfAtom    = SfMNMajorAtom;
};
```

Be minded that `SFVecSize` with `stride 0` **must** be contained in SF Layout. Which will suggest correct element size while go through copy atom creation.

### Layout Deduction

`Xe4BlockScaledConfig` should provide following methods to deduct SF matrix layout.

```
template<int SFVecSize_>
struct Xe4BlockScaledConfig {
  CUTE_HOST_DEVICE static constexpr auto deduce_layoutSFA();
  CUTE_HOST_DEVICE static constexpr auto deduce_layoutSFB();
  
  template < class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFA(ProblemShape problem_shape);

  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFB(ProblemShape problem_shape);
  
  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFA(TiledMma tiled_mma, TileShape_MNK tileshape_mnk);
  
  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFB(TiledMma tiled_mma, TileShape_MNK tileshape_mnk);
} 
```

`deduce_layoutSFA` and `deduce_layoutSFB` are used to deduce the layout for meta tensors in global address space before concrete tensor information is presented to Atom construction APIs.

They return

- `layout_sfa:	((_1,0),(_16,0),(_1,0)):((_1,_1),(_0,0),(_0,0))`
- `layout_sfb:	((_1,0),(_16,0),(_1,0)):((_1,_1),(_0,0),(_0,0))`

`deduce_smem_layoutSFA` and `deduce_smem_layoutSFB` are used for deducing Atom shared memory layouts from `TiledMma` and `TileShape_MNK`. But this layout is not final layout for SF in shared memory, usually user has to add repeat count and other information for further functionalities.

A sample return should include shared memory block layout, tiling layout, in accordance to the relationship between `TiledMma` and `TileShape_MNK`:

- `sfA_layout:	(((_1,_128),(_16,_4)),_1,(_1,_4)):(((_1,_1),(_0,_128)),_0,(_0,_512))`
- `sfB_layout:	(((_1,_256),(_16,_4)),_1,(_1,_4)):(((_1,_1),(_0,_256)),_0,(_0,_1024))`

`tile_atom_to_shape_SFA` and `tile_atom_to_shape_SFB` are used to create global memory layout from concrete scaling factor tensor. The type of return must agree with `deduce_layoutSF*` APIs

### Layout Instance

In order to facilitate design of aforementioned API, list a specific layout instance for a hypothetical block scaled GEMM.

####Basic MMA Tile
- `MmaTileShape`: `(_128, _256, _256)`
- `TiledShape_MNK`: `MmaTileShape`


#### TiledMMA Atom
TiledMMA contains shapes derived from MMA Tile

- `Shape_MNK`: `K` is ranging between 64 to 768, choose a typical value which can form `(_128, _256, _64)` (Pick a small `K` for more general dicussion).

#### Derived Tiling Layouts

Choose Shared Local Memory layout for A/B as `(_8,_256):(_256,_1)` which holds TiledMMA horizontally.

- `mma_shape_A:	((_128,_64),_1,_4,_4)`
- `mma_shape_B:	((_256,_64),_1,_4,_4)`
- `sA_layout:` `((_128,_64),_1,_4,(_1,_4)):((_256,_1),_0,_64,(_0,_32768))`
- `sB_layout:` `((_256,_64),_1,_4,(_1,_4)):((_256,_1),_0,_64,(_0,_65536))`
- `sfA_layout:` `(((_1,_128),(_16,_4)),_1,(_1,_4),_4):(((_1,_1),(_0,_128)),_0,(_0,_512),_2048)`
- `sfB_layout:` `(((_1,_256),(_16,_4)),_1,(_1,_4),_4):(((_1,_1),(_0,_256)),_0,(_0,_1024),_4096)`

`mma_shape_*` suggests that block of `TiledMMA`, `(_128, _64)` for example, uses `K=64` while outer tiler `(_1, _4, _4)` runs 4 unrolled in K direction and uses quadruple buffers.

`s*_Layout` is for shared local memory layouts. Similar to `mma_shape_*`, part of them are block shapes, part of them are for iterations and buffer numbers. Notice that scaling factor layout follows master tensor layout with column major, as well as a `_16` in block size with zero stride.

Depends on which flavours fit the already have APIs. Be minded all shared local memory layout shall be aligned to equal to or larger than **512 bytes**.

As we can see that for every 4 accumulations it requires loading of two scale factors, which doesn't necessarily true for many occation. While the relationship could be changed for better performance, the implementation should accept all possibilities.

## MMA

### MMA Operations

- `XE4_AMMA_FP4FP8` supports `mxfp4` /`nvfp4`/ `nvfp4+`/`mxfp8`
- `XE4_AMMA_MIX` (optional) supports 16-bit floating-point mix with low precision.

In addition to normal A-MMA Operation, MX variants need more information about scaling factors to instantiate template. For example, mxfp4 variant needs `sf_type` and `VS`, with extra sf register types.

```cpp
template <class d_type, class a_type, class b_type, class c_type, class sf_type,
         int M, int N, int K, int VS, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_FP4FP8 {
...
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;
  using SFARegisters = uint32_t[1];
  using SFBRegsiters = uint32_t[1];
...
};
```
- Registers type of scale factor is defined as ``uint32_t[1]`` following code convention but overall it has no significant impact in actual code flow.

- `sf_type` is for scaling factor types which can be `e8m0`, `e4m3` or `e5m3`.

- `VS` is for block size of scaling factor. Supported numbers are `32` and `16`.

- The `fma` function of mx variants accepts extra `Matrix Descriptor` for reflecting instruction form. As an example, fma function sigunature, for groupsync mx variant, should be:

```cpp
  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb
  )
```

Recommand not collapsing MMA Operations into inherited hierarchies, it allows future developers to add new combinations without navigating folded contents.

### MMA Traits/Atom

All AMMA TV Layouts are single lane based and trivial.

```
  using Shape_MNK = Shape<Int<M>,Int<N>,Int<K>>;
  using ThrID   = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
```

Traits of AMMA operation must reflect additional types and constants of underlining operation.

```
  using ValTypeSFA = sf_type;
  using ValTypeSFB = sf_type;
```

Define fragment type of scaling factor in Traits

```
  using FrgTypeSFA = AMMA::smem_sf_desc;
  using FrgTypeSFB = AMMA::smem_sf_desc;
```

In Cutlass example, `MMA_ScaleFactor` is a redefinition of the Traits it reside in with `N` dimension aligned to 128. Follow the convention we should also provide the redefinition.

`MMAControl` member must be initialized with scaling factor information from template parameters. `sf_type` and `VS` provide following combinations to initialize `MMAControl` bits.

```
A_BlockScaleType bits 22:20
0: ue8m0k32. 1: ue8m0k16. 2: ue5m3k32.
3: ue5m3k16. 5: ue4m3k16.

B_BlockScaleType bits 26:24
0: ue8m0k32. 1: ue8m0k16. 2: ue5m3k32.
3: ue5m3k16. 5: ue4m3k16.
```

Traits need to capture `Matrix Descriptor` for scale factor in its members. Prefer `uint32_t` intead of `Matrix Descriptor` type to improve clarity and avoid compiler problems.

```
  // No control
  MMAControl ctrl_ {};

  uint32_t sf_a_;
  uint32_t sf_b_;

  uint64_t* a_barrier_;
  uint64_t* b_barrier_;
```

Some variants only need to capture one SF `Matrix Descriptor` for either A or B.

Traits' ``with`` interface must accept scale factor ``Matrix Descriptor`` to morph itself into more specific variant or to initialize its members. Hence the signature should use `Args/args` instead of `Barriers/barriers`.

```
  template <AMMA::Tracking Method, typename ... Args>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(AMMA::TrackMethod<Method> method, MMAControl ctrl, Args... args) {
    return with(d_type {}, method, ctrl, args...);
  }
```

Recommand not to collapse signature with tuples. Plain list of parameters requires least context memory to use the API.

### SF Fragment Tensor

Like A/B fragment tensor, SF fragment contains ``DescriptorIterator`` instead of raw pointer. Type ``AMMA::smem_sf_desc`` shall inherit from ``DescriptorIterator``, no template parameter needed.

Implementation of `make_fragment_desc` is similar to `make_matrix_desc`. However, it always generate type 3 `Matrix Descriptor` with `Pitch` always use first dimension of input layout. Recommand to put static assert to make sure A/B sf matrics use MN Major.

Implement ``make_tensor`` interface for SF fragment Tensor:

```
template <> struct MakeTensor<AMMA::smem_sf_frag>;
```

Code refactor for handling all Fragment Tensor can be done alone or with the current implementation process.

## Copy

ADMA operator templates have exposed support for all possible combinations of data type which included a version for loading SF matrix. It requires to provide type 3 Matrix Descriptor which is reserved for SF matrix (MX metadata). A’s MX metadata is always in an m-major and B’s MX metadata is always in an n-major.

### Copy Traits/Atom

Add API set ``make_adma_atom_*_xe4`` on top of ``make_tma_copy_*_xe4`` with extra cluster tiler. APIs ``make_adma_atom_*_xe4`` can be used for creating Copy Atoms for loading both A/B matrix and their SF matrix.

Caveat: In a performant implementation, it is better to load all scale factors once for all K blocks than to load them in each K iteration. Hence, ``make_adma_atom_*_xe4`` need to accept untiled layout, not just layout forms return from ``tile_to_shape`` or ``tile_to_mma_shape``.

Caveat: Also, the data type for invoking APIs for creating SF Copy Atoms tends to be uint16_t instead of scale types, like e5m3.

#### Layout TensorDescriptor and MatrixDescriptor

Continue from MMA layout example we follow concrete example. All four layouts will be the input to ``make_adma_atom_*_xe4``

```
TiledMMA
  ThrLayoutVMNK:  (_1,_1,_1,_1):(_0,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:      _1:_0
  Shape_MNK:  (_128,_256,_64)
  LayoutA_TV: (_1,(_128,_64)):(_0,(_1,_128))
  LayoutB_TV: (_1,(_256,_64)):(_0,(_1,_256))
  LayoutC_TV: (_1,(_128,_256)):(_0,(_1,_128))
```
- `mma_tiler:` `(_128,_256,_256)`
- `sA_layout:` `((_128,_64),_1,_4,(_1,_4)):((_256,_1),_0,_64,(_0,_32768))`
- `sB_layout:` `((_256,_64),_1,_4,(_1,_4)):((_256,_1),_0,_64,(_0,_65536))`
- `sfA_layout:` `(((_1,_128),(_16,_4)),_1,(_1,_4),_4):(((_1,_1),(_0,_128)),_0,(_0,_512),_2048)`
- `sfB_layout:` `(((_1,_256),(_16,_4)),_1,(_1,_4),_4):(((_1,_1),(_0,_256)),_0,(_0,_1024),_4096)`

Also, Matrix from global memory will have simple row major layout with runtime dimension.

- `A:` `(512, 2048) : (2048, 1)`
- `B:` `(1024, 2048) : (2048, 1)`

Scaling factor layout from API `tile_atom_to_shape_SF*` with global memory tensors should be:

- `layout_SFA:	((_1,512),(_16,128),(_1,1)):((_1,_1),(_0,512),(_0,65536))`
- `layout_SFB:	((_1,1024),(_16,128),(_1,1)):((_1,_1),(_0,1024),(_0,131072))`

The already have API for sm90/sm100 will generate correct global shape and shared memory box size.

- `A Tensor Descriptor` `(256, 128, 1, 1, 1) : (1, 2048, 256, 1, 1)`
- `A smem Box shape` `(256, 128, 1, 1, 1)`

The first dimension of `tma_gbasis` is always the true stride of shared memory layout. And first dimension of smem Box shape reflect the first dimension. Hence we should simply use the number (`256`) as Pitch in MatrixDescriptor, and clean up extra code for infer shared memory Pitch.


### SF TMA Tensor

## Example API usage by MXFP8 GEMM

Original cutlass doesn't provide simple example for block scale gemm. In order for quick ramp up and verification, we could provide one based on Pre-Si reference example [mxfp-bf8_gemm](https://github.com/intel-sandbox/drivers.gpu.compute.workloads/blob/main/simt_workloads/examples/01_mxfp_gemm/bf8_gemm.cpp)

## Collectives
### Dispatching tag for Gemm Collective Builder

Reuse type ``OpClassBlockScaledTensorOp`` for dispatching Collective Builder, Builder specialization can be differentiated by Architecture Tag ``Xe4``.

### Implement Xe4 Block Scaled Gemm Collective Builder

Implement ``CollectiveBuilder`` in ``xe4_blockscaled_amma_builder.inl``.

### Dispatching tag for CollectiveMMA

Implement ``MainloopXe4AdmaAmmaWarpSpecializedBlockScaled`` for dispatching CollectiveMMA template.

### Implement Xe4 CollectiveMMA

Implmment ``CollectiveMma`` in ``xe4_blockscaled_mma_warpspecialized.hpp``.

