/**
 * @file ArchitectureList.ixx
 * @brief Architectures known at compile time, found by the name a manifest gives them.
 */

export module Dnn.Models.ArchitectureList;

import std;

namespace Mila::Dnn
{
    /**
     * @brief A fixed list of architecture descriptors, searched by name.
     *
     * No registration and no run-time table: an entry missing a member its caller uses is a compile error, and a
     * name not in the list is the caller's to refuse (ModelHandle.md 3.3).
     */
    export template<typename... TArchitectures>
    struct ArchitectureList
    {
        /**
         * @brief Call the visitor with `std::type_identity<A>` for the architecture A that has this name.
         *
         * @return False when no architecture in the list has it; the visitor is then not called.
         */
        template<typename TVisitor>
        static bool visit( std::string_view name, TVisitor&& visitor )
        {
            return ( visitIfNamed<TArchitectures>( name, visitor ) || ... );
        }

        /// Call the visitor with `std::type_identity<A>` for every architecture, in list order.
        template<typename TVisitor>
        static void forEach( TVisitor&& visitor )
        {
            ( visitor( std::type_identity<TArchitectures>{} ), ... );
        }

        static constexpr bool contains( std::string_view name ) noexcept
        {
            return ( ( name == TArchitectures::name ) || ... );
        }

        /// The names in list order, for a refusal: "gemma, llama, qwen".
        static std::string names()
        {
            std::string listed;

            ( ( listed += ( listed.empty() ? "" : ", " ) + std::string( TArchitectures::name ) ), ... );

            return listed;
        }

    private:

        template<typename TArchitecture, typename TVisitor>
        static bool visitIfNamed( std::string_view name, TVisitor& visitor )
        {
            if ( name != TArchitecture::name )
                return false;

            visitor( std::type_identity<TArchitecture>{} );

            return true;
        }
    };
}
