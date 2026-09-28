/**
 * @file Chat.ModelPrecision.ixx
 * @brief The compute precision a Chat session runs at.
 */

export module Chat.ModelPrecision;

namespace Mila::ChatApp
{
    export enum class ModelPrecision
    {
        FP32,
        BF16
    };
}
