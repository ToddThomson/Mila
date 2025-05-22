/**
 * @file RandomGenerator.ixx
 * @brief Provides a centralized random number generator for the Mila library.
 */

module;
#include <random>
#include <mutex>

export module Core.RandomGenerator;

namespace Mila::Core
{
    /**
     * @brief Singleton class providing centralized random number generation.
     *
     * This class manages random number generation for the entire codebase,
     * allowing control over reproducibility through seed management.
     */
    export class RandomGenerator {
    public:
        /**
         * @brief Gets the singleton instance of the RandomGenerator.
         *
         * @return Reference to the singleton instance
         */
        static RandomGenerator& getInstance() {
            static RandomGenerator instance;
            return instance;
        }

        /**
         * @brief Sets the global random seed.
         *
         * @param seed The seed value (use 0 for non-deterministic behavior from std::random_device)
         */
        void setSeed( unsigned int seed ) {
            std::lock_guard<std::mutex> lock( mutex_ );
            if ( seed == 0 ) {
                std::random_device rd;
                seed_ = rd();
            }
            else {
                seed_ = seed;
            }
            generator_ = std::mt19937( seed_ );
        }

        /**
         * @brief Gets the currently set random seed.
         *
         * @return The currently used seed value
         */
        unsigned int getSeed() const {
            std::lock_guard<std::mutex> lock( mutex_ );
            return seed_;
        }

        /**
         * @brief Gets a random number generator initialized with the global seed.
         *
         * This returns a copy of the generator, so subsequent calls to setSeed
         * won't affect already obtained generators.
         *
         * @return A copy of the random number generator
         */
        std::mt19937 getGenerator() const {
            std::lock_guard<std::mutex> lock( mutex_ );
            return generator_;
        }

    private:
        RandomGenerator() {
            // Initialize with a non-deterministic seed by default for production use
            std::random_device rd;
            seed_ = rd();
            generator_ = std::mt19937( seed_ );
        }

        // Prevent copying
        RandomGenerator( const RandomGenerator& ) = delete;
        RandomGenerator& operator=( const RandomGenerator& ) = delete;

        mutable std::mutex mutex_;
        unsigned int seed_;
        std::mt19937 generator_;
    };
}