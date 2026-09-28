/**
 * @file Policies.ixx
 * @brief Re-exports every weight quantization policy and the concepts that classify them.
 */

export module Dnn.Quantization.Weight.Policies;

export import Dnn.Quantization.Weight.WeightQuantPolicy;
export import Dnn.Quantization.Weight.NoWeightQuant;
export import Dnn.Quantization.Weight.PerChannelFp8;
export import Dnn.Quantization.Weight.PerGroupInt4;
export import Dnn.Quantization.Weight.PerGroupFp4;
export import Dnn.Quantization.Weight.PerGroupCodebook2;
export import Dnn.Quantization.Weight.PerGroupCodebook3;
export import Dnn.Quantization.Weight.HasCodebookTable;
export import Dnn.Quantization.Weight.HasHighBitPlane;

namespace Mila::Dnn::Quant::Weight
{
    // The companion traits must select exactly the policies that declare them: a codebook policy allocating
    // no table, or an FP4 policy allocating one, would both be silent.
    static_assert( HasCodebookTable<PerGroupCodebook2<>> );
    static_assert( HasCodebookTable<PerGroupCodebook3<>> );
    static_assert( !HasCodebookTable<PerGroupFp4<>> );
    static_assert( !HasCodebookTable<PerGroupInt4<>> );
    static_assert( !HasCodebookTable<NoWeightQuant> );
    static_assert( HasHighBitPlane<PerGroupCodebook3<>> );
    static_assert( !HasHighBitPlane<PerGroupCodebook2<>> );
    static_assert( !HasHighBitPlane<PerGroupFp4<>> );
    static_assert( !HasHighBitPlane<PerGroupInt4<>> );
}
