/**
 * @file Chat.MemoryHolder.ixx
 * @brief Chat's hold on the model's memory: locked while it is in use, let go five minutes after the last turn.
 */

export module Chat.MemoryHolder;

import std;
import Mila;

namespace Mila::ChatApp
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Locks the device's memory for a turn and lets it go once Chat has been idle for the backstop.
     *
     * A program in a terminal is never told whether its window is the one in use, so the time since the last turn is
     * Chat's only release (Specifications/Deployment.md 12.10, Todd, 2026-11-13). The release runs on its own thread and
     * touches only the device, whose hold covers whatever model is open on it, so a model switched meanwhile is no
     * concern of it. On a device whose memory is never paged every hold is the same, and this does nothing visible.
     */
    export class MemoryHolder
    {
    public:

        static constexpr std::chrono::minutes kBackstop{ 5 };

        MemoryHolder()
            : releaser_( [this]( std::stop_token stop ) { releaseWhenIdle( stop ); } )
        {
        }

        MemoryHolder( const MemoryHolder& ) = delete;
        MemoryHolder& operator=( const MemoryHolder& ) = delete;

        /// Lock the memory of the device `device_id` names, until `releaseAfterBackstop` and the backstop after it.
        void hold( DeviceId device_id )
        {
            std::shared_ptr<Device> device = DeviceRegistry::instance().getDevice( device_id );

            {
                // Under the lock, so a release that fell due meanwhile cannot undo it.
                std::lock_guard lock( mutex_ );
                release_at_.reset();
                device_ = std::move( device );
                device_->setMemoryHold( MemoryHold::Locked );
            }

            wake_.notify_all();
        }

        /// The model is idle from now: its memory is let go after the backstop, unless a turn holds it again first.
        void releaseAfterBackstop()
        {
            {
                std::lock_guard lock( mutex_ );
                release_at_ = std::chrono::steady_clock::now() + kBackstop;
            }

            wake_.notify_all();
        }

    private:

        void releaseWhenIdle( std::stop_token stop )
        {
            std::unique_lock lock( mutex_ );

            while ( !stop.stop_requested() )
            {
                if ( !release_at_ )
                {
                    wake_.wait( lock, stop, [this] { return release_at_.has_value(); } );

                    continue;
                }

                const auto due = *release_at_;

                // A turn that holds again, or ends again, moves or clears the time; the wait starts over.
                if ( wake_.wait_until( lock, stop, due, [this, due] { return release_at_ != due; } ) || stop.stop_requested() )
                {
                    continue;
                }

                release_at_.reset();

                if ( device_ )
                {
                    device_->setMemoryHold( MemoryHold::None );
                }
            }
        }

        std::mutex mutex_;
        std::condition_variable_any wake_;
        std::optional<std::chrono::steady_clock::time_point> release_at_;
        std::shared_ptr<Device> device_;

        // Last, so it stops before what it reads is destroyed.
        std::jthread releaser_;
    };
}
