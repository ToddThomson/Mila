/**
 * @file Chat.ModelType.ixx
 * @brief The model family, as the library names it.
 */

export module Chat.ModelType;

import Mila;

namespace Mila::ChatApp
{
    // The library's enum, so the family an installed model reports is the one Chat switches on. Chat runs the
    // instruct families -- Gemma, Llama and Qwen -- and refuses the rest at the catalogue.
    export using ModelType = Mila::Dnn::ModelType;
}
