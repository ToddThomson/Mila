/**
 * @file Chat.ModelNotFound.ixx
 * @brief A model name, given as an instruction, that named nothing the store holds.
 */

export module Chat.ModelNotFound;

import std;

namespace Mila::ChatApp
{
    /// A model name, given as an instruction, that named nothing the store holds.
    export struct ModelNotFound : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
}
