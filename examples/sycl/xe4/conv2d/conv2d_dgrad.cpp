#include <sycl/sycl.hpp>
#include "conv2d_validation.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/mma_xe4.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/conv/kernel/xe4_implicit_gemm_dma_warpspecialized.hpp"

using namespace sycl;
using namespace cutlass::xe4;
using namespace cute;
using namespace cute::detail;
using namespace cute::xe4;
using namespace cutlass::conv::collective;
using namespace cutlass::epilogue::collective;

class DGRAD_WITH_PAD_WITH_DILATION;
class DGRAD_PERFORMANCE_NORMAL;
class DGRAD_PERFORMANCE_NHW_NO_MULTI_WGM;
class DGRAD_PERFORMANCE_UNET0;
class DGRAD_PERFORMANCE_UNET1;

template <class IntT>
CUTLASS_HOST_DEVICE
cute::Stride<cute::Stride<IntT, IntT, IntT>, cute::Int<1>>
make_cute_packed_stride(
    cute::Stride<cute::Stride<IntT, IntT, IntT>, cute::Int<1>> s,
    cute::array<IntT, 4> stride_nhwc,
    cutlass::conv::Operator ConvOp) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  assert(stride_nhwc[3] == 1);
  auto s_copy = s;
  cute::for_each(cute::make_seq<3>{}, [&](auto i) {
    cute::get<0,i>(s_copy) = stride_nhwc[2-i];
  });
  return s_copy;
}

template<typename test, class wgM, class wgN, class wgK>
int run_test(const conv2d::problem_shape_t &problem_shape)
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    // input:   (N, H, W, C)
    // kernel:  (K, R, S, C)
    // output:  (Out_N, Out_H, Out_W, Out_C)
    uint32_t N = problem_shape.get_in_batch();
    uint32_t H = problem_shape.get_in_height();
    uint32_t W = problem_shape.get_in_width();
    uint32_t C = problem_shape.get_in_channel();
    uint32_t K = problem_shape.get_kernel_num();
    uint32_t R = problem_shape.get_kernel_height();
    uint32_t S = problem_shape.get_kernel_width();
    uint32_t Out_N = problem_shape.get_out_batch();
    uint32_t Out_H = problem_shape.get_out_height();
    uint32_t Out_W = problem_shape.get_out_width();
    uint32_t Out_C = problem_shape.get_out_channel();
    uint32_t padding_top = problem_shape.get_padding_top();
    uint32_t padding_left = problem_shape.get_padding_left();
    uint32_t padding_bottom = problem_shape.get_padding_bottom();
    uint32_t padding_right = problem_shape.get_padding_right();
    uint32_t stride_h = problem_shape.get_stride_h();
    uint32_t stride_w = problem_shape.get_stride_w();
    uint32_t dilation_h = problem_shape.get_dilation_h();
    uint32_t dilation_w = problem_shape.get_dilation_w();

    using ElementAct = fp16;
    using ElementFlt = fp16;
    using ElementAcc = float;
    using ElementOut = fp16;

    constexpr uint32_t Stages = 3;

    auto bM = wgM{};
    auto bN = wgN{};
    auto bK = wgK{};
    static_assert(bM % LANESIZE == 0);
    using TileShapeMNK = Shape<Int<bM>, Int<bN>, Shape<Int<bK>>>;
    using ClusterShapeMNK = Shape<_1, _1, _1>;

    // for dgrad, shapeA: NPQK, shapeB: CRSK, shapeC:NHWC
    uint32_t sizeA = Out_C * Out_W * Out_H * Out_N;
    uint32_t sizeB = K * S * R * C;
    uint32_t sizeC = C * W * H * N;

    auto A_shared = malloc_shared<ElementAct>(sizeA, q);
    std::generate_n(A_shared, sizeA, [=]() { return random_float(); });
    // std::iota(A_shared, A_shared + sizeA, 0);

    auto B_shared = malloc_shared<ElementFlt>(sizeB, q);
    std::generate_n(B_shared, sizeB, [=]() { return random_float(); });

    auto C_shared = malloc_shared<ElementOut>(sizeC, q);
    std::fill_n(C_shared, sizeC, ElementOut(0));

    constexpr uint32_t SubGroupSize = 32;
    constexpr uint32_t MmaSubgroupNum = 1;
    constexpr uint32_t SchedSubgroupNum = 1;
    constexpr uint32_t LoadSubgroupNum = 1;
    constexpr uint32_t StoreSubgroupNum = 1;
    constexpr uint32_t NumControlSubGroup = MmaSubgroupNum + SchedSubgroupNum + LoadSubgroupNum + StoreSubgroupNum;
    constexpr uint32_t NumPostOpSubGroup = 0;
    range<3> local_range(1, NumControlSubGroup + NumPostOpSubGroup, SubGroupSize);
    // the shape of mat_m is NHW * RSK and mat_n is RSK * C
    uint32_t mat_m = W * H * N;
    uint32_t mat_n = C;
    static constexpr bool is_persistent_mode = true;
    static constexpr uint32_t num_xecore_x = 1;
    static constexpr uint32_t num_xecore_y = 1;
    range<3> group_range(1, num_xecore_y, num_xecore_x);
    if (!is_persistent_mode) {
        group_range = range<3>(1, ceil_div(mat_m, bM), ceil_div(mat_n, bN));
    }
    std::cout << "IsPersistentMode: " << is_persistent_mode << std::endl;
    std::cout << "Group range: {" << group_range[0] << ", " << group_range[1] << ", "
        << group_range[2]  << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
        cutlass::arch::Xe4, cutlass::arch::OpClassTensorOp,
        cutlass::conv::Operator::kDgrad,
        ElementAct, cutlass::layout::TensorNHWC, 8,
        ElementFlt, cutlass::layout::TensorNHWC, 8,
        tuple<ElementAcc, ElementOut>,
        TileShapeMNK, ClusterShapeMNK,
        cutlass::conv::collective::StageCount<static_cast<int>(Stages)>,
        cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;
    using ProblemShape = cutlass::conv::ConvProblemShape<
                                    cutlass::conv::Operator::kDgrad,
                                    CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
    ProblemShape cutlass_problem_shape {
        cutlass::conv::Mode::kCrossCorrelation,
        {static_cast<int>(N), static_cast<int>(H), static_cast<int>(W), static_cast<int>(C)},   // NHWC
        {static_cast<int>(K), static_cast<int>(R), static_cast<int>(S), static_cast<int>(C)},   // KRSC
        {static_cast<int>(padding_top), static_cast<int>(padding_left)}, // padding lower (pad_top, pad_left)
        {static_cast<int>(padding_bottom), static_cast<int>(padding_right)},   // padding upper (pad_bottom, pad_right)
        {static_cast<int>(stride_h), static_cast<int>(stride_w)},           // stride (stride_h, stride_w)
        {static_cast<int>(dilation_h), static_cast<int>(dilation_w)},       // dilation (dilation_h, dilation_w)
        1   // group
    };
    static constexpr int StagesC = 1;
    static constexpr int StagesD = 1;
    static constexpr bool ReuseSmemC = false;
    static constexpr bool DelayTmaStore = false;
    static constexpr int NumControlWarps = 4;
    static constexpr int NumEpilogueWarps = 16;
    static constexpr int NumElementsPerThread = 32;
    static constexpr int FragmentSize = 32 / sizeof(ElementOut);

    using TileShape_MN = decltype(select<0,1>(TileShapeMNK{}));
    using SmemLayoutAtomD = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));
    using CollectiveEpilogue = EpilogueConv<
        StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumControlWarps, NumEpilogueWarps,
        TileShapeMNK,
        void, void, void,
        ElementOut, 
        cutlass::detail::TagToStrideC_t<cutlass::layout::TensorNHWC>,
        void, void, void, void, void,
        xe4::ASYNC_ROW_STORE_IM2COL<slm_matrix_type::type1>,
        SmemLayoutAtomD, 
        void, void
    >;

    using StrideC = decltype(cute::Stride<cute::Stride<int64_t, int64_t, int64_t>,cute::Int<1>>{});
    auto stride_D = append<3>(make_cute_packed_stride(StrideC{}, cutlass_problem_shape.stride_C, cutlass::conv::Operator::kDgrad), _0{});
    using ConvKernel = cutlass::conv::kernel::Xe4ConvUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
        auto args = typename ConvKernel::Arguments {
            cutlass_problem_shape,
            {A_shared, B_shared},
            {C_shared, stride_D}
        };

        ConvKernel kernel;
        auto params = kernel.to_underlying_arguments(args, nullptr);
        kernel(params);
    }).wait();

    uint32_t err_cnt = validate_conv2d_result_by_dgrad_onednn(A_shared, B_shared, C_shared, problem_shape);

    int rtn = 0;
    if (err_cnt > 0)
    {
        std::cout << "Test Failed!" << std::endl;
        rtn = -1;
    }
    else
    {
        std::cout << "Test Pass!" << std::endl;
        rtn = 0;
    }

    return rtn;
}

int main(){
#if defined(TEST_WITH_PAD_WITH_DILATION)
    conv2d::problem_shape_t with_pad_with_dilation {{512, 16, 16, 2}, {512, 5, 2, 256}, {1, 0}, {0, 1}, {1, 1}, {2, 3}};
    run_test<DGRAD_WITH_PAD_WITH_DILATION, Int<256>, Int<512>, Int<128>>(with_pad_with_dilation);
#elif defined(TEST_PERFORMANCE_NORMAL)
    conv2d::problem_shape_t performance_normal {{256, 16, 16, 4}, {256, 3, 3, 512}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    run_test<DGRAD_PERFORMANCE_NORMAL, Int<256>, Int<256>, Int<128>>(performance_normal);
#elif defined(TEST_PERFORMANCE_NHW_NO_MULTI_WGM)
    conv2d::problem_shape_t performance_nhw_no_multi_wgm {{256, 28, 28, 4}, {256, 3, 3, 512}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    run_test<DGRAD_PERFORMANCE_NHW_NO_MULTI_WGM, Int<256>, Int<256>, Int<128>>(performance_nhw_no_multi_wgm);
#elif defined(TEST_PERFORMANCE_UNET0)
    conv2d::problem_shape_t performance_unet0 {{128, 256, 256, 1}, {128, 3, 3, 128}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    run_test<DGRAD_PERFORMANCE_UNET0, Int<256>, Int<128>, Int<128>>(performance_unet0);
#elif defined(TEST_PERFORMANCE_UNET1)
    conv2d::problem_shape_t performance_unet1 {{512, 64, 64, 1}, {512, 3, 3, 512}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    run_test<DGRAD_PERFORMANCE_UNET1, Int<256>, Int<512>, Int<128>>(performance_unet1);
#endif

    return 0;
}