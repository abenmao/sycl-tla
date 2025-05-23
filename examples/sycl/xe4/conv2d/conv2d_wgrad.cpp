#include <sycl/sycl.hpp>
#include "conv2d_validation.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/arch/mma_xe4.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/conv/kernel/xe4_implicit_gemm_dma_warpspecialized.hpp"

using namespace sycl;
using namespace cutlass::xe4;
using namespace cute;
using namespace cute::detail;
using namespace cute::xe4;
using namespace cutlass::conv::collective;
using namespace cutlass::epilogue::collective;

class WGRAD_WITH_PAD_WITH_STRIDE_WITH_DILATION;
class WGRAD_PERFORMANCE_NORMAL;
class WGRAD_PERFORMANCE_UNET;

template<typename test, class wgM, class wgN, class wgK>
int run_test(const conv2d::problem_shape_t &problem_shape)
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    // input:   (N, Out_H, Out_W, Out_C)
    // kernel:  (N, H, W, C)
    // output:  (Out_C, R, S, C)
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
    using TileShapeMNK = Shape<Int<bM>, Shape<Int<bN>>, Shape<Int<bK>>>;
    using ClusterShapeMNK = Shape<_1, _1, _1>;

    // for wgrad, shapeA: NPQK, shapeB: NHWC, shapeC:KRSC
    uint32_t sizeA = Out_C * Out_W * Out_H * Out_N;
    uint32_t sizeB = C * W * H * N;
    uint32_t sizeC = C * S * R * K;

    auto A_shared = malloc_shared<ElementAct>(sizeA, q);
    std::generate_n(A_shared, sizeA, [=]() { return random_float(); });

    auto B_shared = malloc_shared<ElementFlt>(sizeB, q);
    std::generate_n(B_shared, sizeB, [=]() { return random_float(); });

    auto C_shared = malloc_shared<ElementOut>(sizeC, q);
    std::fill_n(C_shared, sizeC, ElementOut(0));

    constexpr int NumControlWarps = 4;
    constexpr int NumEpilogueWarps = 16;
    range<3> local_range(1, NumControlWarps + NumEpilogueWarps, cutlass::NumThreadsPerWarp);
    // the shape of mat_m is K * (NPQ) and mat_n is (NPQ) * (RSC)
    uint32_t mat_m = Out_C;
    static constexpr bool is_persistent_mode = true;
    static constexpr uint32_t num_xecore_x = 1;
    static constexpr uint32_t num_xecore_y = 1;
    range<3> group_range(1, num_xecore_y, num_xecore_x);
    if (!is_persistent_mode) {
        group_range = range<3>(1, ceil_div(mat_m, bM), R * S * ceil_div(C, bN));
    }
    std::cout << "IsPersistentMode: " << is_persistent_mode << std::endl;
    std::cout << "Group range: {" << group_range[0] << ", " << group_range[1] << ", "
        << group_range[2]  << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
        cutlass::arch::Xe4, cutlass::arch::OpClassTensorOp,
        cutlass::conv::Operator::kWgrad,
        ElementAct, cutlass::layout::TensorNHWC, 8,
        ElementFlt, cutlass::layout::TensorNHWC, 8,
        tuple<ElementAcc, ElementOut>,
        TileShapeMNK, ClusterShapeMNK,
        cutlass::conv::collective::StageCount<static_cast<int>(Stages)>,
        cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;
    using ProblemShape = cutlass::conv::ConvProblemShape<
                                    cutlass::conv::Operator::kWgrad,
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

    // Build the epilogue
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Xe4, cutlass::arch::OpClassTensorOp,
        TileShapeMNK, ClusterShapeMNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc, 
        void, cutlass::layout::TensorKCSR, 512,
        ElementOut, cutlass::layout::TensorKCSR, 512,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        cutlass::epilogue::fusion::EltAct<cutlass::epilogue::thread::Identity, ElementOut, ElementOut>
    >::CollectiveOp;

    using StrideC = TagToStrideC<cutlass::layout::TensorKCSR>::type;
    auto stride_D = cutlass::make_cute_packed_stride(StrideC{}, cutlass_problem_shape.shape_C, cutlass_problem_shape.stride_C, cutlass::conv::Operator::kWgrad);
    using ConvKernel = cutlass::conv::kernel::Xe4ConvUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
        using FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;
        auto callbacks_args = typename FusionCallbacks::Arguments {};
        auto args = typename ConvKernel::Arguments {
            cutlass_problem_shape,
            {A_shared, B_shared},
            {callbacks_args, nullptr, stride_D, C_shared, stride_D}
        };

        ConvKernel kernel;
        auto params = kernel.to_underlying_arguments(args, nullptr);
        kernel(params);
    }).wait();

    uint32_t err_cnt = validate_conv2d_result_by_wgrad_onednn(A_shared, B_shared, C_shared, problem_shape);

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
#if defined(TEST_WITH_PAD_WITH_STRIDE_WITH_DILATION)
    conv2d::problem_shape_t with_pad_with_stride_with_dilation {{32, 15, 16, 2}, {32, 5, 2, 256}, {1, 0}, {0, 1}, {2, 3}, {2, 3}};
    run_test<WGRAD_WITH_PAD_WITH_STRIDE_WITH_DILATION, Int<256>, Int<512>, Int<128>>(with_pad_with_stride_with_dilation);
#elif defined(TEST_PERFORMANCE_NORMAL)
    conv2d::problem_shape_t performance_normal {{256, 16, 16, 4}, {256, 3, 3, 512}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    run_test<WGRAD_PERFORMANCE_NORMAL, Int<256>, Int<512>, Int<128>>(performance_normal);
#elif defined(TEST_PERFORMANCE_UNET)
    conv2d::problem_shape_t performance_unet {{128, 256, 256, 1}, {128, 3, 3, 128}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    run_test<WGRAD_PERFORMANCE_UNET, Int<128>, Int<512>, Int<128>>(performance_unet);
#endif

    return 0;
}