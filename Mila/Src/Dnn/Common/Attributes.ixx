/**
 * @file Attributes.ixx
 * @brief CRTP base class for operation attributes in the deep learning framework
 */

module;
#include <memory>
#include <unordered_map>

export module Dnn.Attributes;

namespace Mila::Dnn
{
    /**
     * @brief CRTP base class for attribute containers
     *
     * This template serves as a base for operation-specific attribute containers,
     * allowing each operation to define its own attribute enum while providing
     * common tensor storage and retrieval functionality.
     *
     * @tparam Derived The derived class that defines the attribute enum
     */
    export template <typename Derived>
    class Attributes {
    };
}