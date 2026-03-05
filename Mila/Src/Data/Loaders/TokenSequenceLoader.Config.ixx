module;
#include <cstddef>
#include <cstdint>

export module Data.TokenSequenceLoader:Config;

namespace Mila::Data
{
    /**
     * @brief Configuration for StreamingSequenceLoader behavior.
     */
    export struct TokenSequenceLoaderConfig
    {
        /**
         * @brief Size of token window to load from disk (in tokens).
         *
         * Set to 0 for automatic sizing based on memory constraints.
         * Larger windows reduce I/O frequency but increase memory usage.
         *
         * Default: 0 (automatic, typically ~25M tokens)
         */
        size_t token_window_size = 0;

        /**
         * @brief Timeout for initial batch preparation (milliseconds).
         *
         * How long to wait for the first batch during construction/reset.
         *
         * Default: 5000ms (5 seconds)
         */
        uint32_t initialization_timeout_ms = 5000;

        /**
         * @brief Timeout for subsequent batch preparation (milliseconds).
         *
         * How long to wait for each batch in nextBatch().
         * Should be generous enough to account for disk I/O variance.
         *
         * Default: 10000ms (10 seconds)
         */
        uint32_t batch_timeout_ms = 10000;

        /**
         * @brief Enable verbose logging during initialization and operation.
         *
         * When true, prints dataset statistics, window sizes, and batch counts.
         *
         * Default: false
         */
        bool verbose_logging = false;

        /**
         * @brief DEPRECATED: No longer used in refactored implementation.
         *
         * The new architecture uses double buffering instead of a queue,
         * so this parameter has no effect.
         */
        [[deprecated("Queue-based architecture replaced with double buffering")]]
        size_t max_queue_size = 2;
    };

}