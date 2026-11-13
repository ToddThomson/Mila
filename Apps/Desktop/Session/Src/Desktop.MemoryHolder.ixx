/**
 * @file Desktop.MemoryHolder.ixx
 * @brief Mila Desktop's hold on the model's memory: locked while Mila is the application in use, let go when it is not.
 */

export module Desktop.MemoryHolder;

import std;
import Mila;

namespace Mila::Desktop
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Locks the device's memory while Mila is in use, and lets it go in the background or after the backstop.
     *
     * The window says when Mila becomes the application in use and when it stops being it, which includes the screen
     * locking and the Mac going to sleep; a turn holds the memory too, and its end starts the backstop, the release for
     * a window left in front and unused (Specifications/Deployment.md 12.10, Todd, 2026-11-13). A reply under way when
     * the person switches away completes: letting go of a lock takes nothing the GPU is using. Thread-safe: the window's
     * messages arrive on the reading thread, turns on the session's. On a device whose memory is never paged every hold
     * is the same, and this does nothing visible.
     */
    export class MemoryHolder
    {
    public:

        static constexpr std::chrono::minutes kBackstop{ 30 };

        MemoryHolder()
            : releaser_( [this]( std::stop_token stop ) { releaseWhenIdle( stop ); } )
        {
        }

        MemoryHolder( const MemoryHolder& ) = delete;
        MemoryHolder& operator=( const MemoryHolder& ) = delete;

        /// A turn starts, or a model has just loaded, on the device `device_id` names: its memory is locked.
        void use( DeviceId device_id )
        {
            std::shared_ptr<Device> device = DeviceRegistry::instance().getDevice( device_id );
            std::lock_guard lock( mutex_ );

            device_ = std::move( device );
            release_at_.reset();
            device_->setMemoryHold( MemoryHold::Locked );
        }

        /// The model is idle from now: held for the backstop while Mila is in use, let go at once when it is not.
        void idle()
        {
            {
                std::lock_guard lock( mutex_ );

                if ( !active_ )
                {
                    releaseNow();

                    return;
                }

                release_at_ = std::chrono::steady_clock::now() + kBackstop;
            }

            wake_.notify_all();
        }

        /// Mila became the application in use: the memory is locked again, and the backstop runs from now.
        void activate()
        {
            {
                std::lock_guard lock( mutex_ );
                active_ = true;

                if ( !device_ )
                {
                    return;
                }

                device_->setMemoryHold( MemoryHold::Locked );
                release_at_ = std::chrono::steady_clock::now() + kBackstop;
            }

            wake_.notify_all();
        }

        /// Mila is no longer the application in use: the memory is let go now.
        void deactivate()
        {
            std::lock_guard lock( mutex_ );
            active_ = false;
            releaseNow();
        }

    private:

        // Called with the lock held, so a hold and a release never cross.
        void releaseNow()
        {
            release_at_.reset();

            if ( device_ )
            {
                device_->setMemoryHold( MemoryHold::None );
            }
        }

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

                // A turn, or the window, moves or clears the time; the wait starts over.
                if ( wake_.wait_until( lock, stop, due, [this, due] { return release_at_ != due; } ) || stop.stop_requested() )
                {
                    continue;
                }

                releaseNow();
            }
        }

        std::mutex mutex_;
        std::condition_variable_any wake_;
        std::optional<std::chrono::steady_clock::time_point> release_at_;
        std::shared_ptr<Device> device_;

        // The window starts in front.
        bool active_{ true };

        // Last, so it stops before what it reads is destroyed.
        std::jthread releaser_;
    };
}
