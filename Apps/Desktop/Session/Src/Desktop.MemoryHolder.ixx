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
     * the person switches away completes: letting go of a lock takes nothing the GPU is using. On a device whose memory
     * is never paged every hold is the same, and this does nothing visible.
     *
     * Every lock and release is made on this holder's own thread: a lock that must first bring memory back from the
     * compressor or swap takes seconds, and the window's messages must not wait on it. Only a turn waits, for its own
     * lock. Such a lock reports its progress as it goes: `progress( done, total )` in bytes, from 0 to `total`.
     * Thread-safe: the window's messages arrive on the reading thread, turns on the session's.
     */
    export class MemoryHolder
    {
    public:

        static constexpr std::chrono::minutes kBackstop{ 30 };

        using Progress = std::function<void( std::size_t done, std::size_t total )>;

        explicit MemoryHolder( Progress progress )
            : progress_( std::move( progress ) ), worker_( [this]( std::stop_token stop ) { work( stop ); } )
        {
        }

        MemoryHolder( const MemoryHolder& ) = delete;
        MemoryHolder& operator=( const MemoryHolder& ) = delete;

        /**
         * @brief A turn starts, or a model has just loaded, on the device `device_id` names: its memory is locked.
         *
         * Returns once it is, or once Mila stopped being the application in use while it waited.
         */
        void use( DeviceId device_id )
        {
            std::shared_ptr<Device> device = DeviceRegistry::instance().getDevice( device_id );
            std::unique_lock lock( mutex_ );

            device_ = std::move( device );
            release_at_.reset();
            want( MemoryHold::Locked );

            applied_changed_.wait( lock, [this]
                {
                    return wanted_ != MemoryHold::Locked || ( applied_ == MemoryHold::Locked && applied_device_ == device_ );
                } );
        }

        /// The model is idle from now: held for the backstop while Mila is in use, let go at once when it is not.
        void idle()
        {
            std::lock_guard lock( mutex_ );

            if ( !active_ )
            {
                release_at_.reset();
                want( MemoryHold::None );

                return;
            }

            release_at_ = std::chrono::steady_clock::now() + kBackstop;
            changed();
        }

        /// Mila became the application in use: the memory is locked again, and the backstop runs from now.
        void activate()
        {
            std::lock_guard lock( mutex_ );
            active_ = true;

            if ( !device_ )
            {
                return;
            }

            release_at_ = std::chrono::steady_clock::now() + kBackstop;
            want( MemoryHold::Locked );
        }

        /// Mila is no longer the application in use: the memory is let go now.
        void deactivate()
        {
            std::lock_guard lock( mutex_ );
            active_ = false;
            release_at_.reset();
            want( MemoryHold::None );
        }

    private:

        static constexpr std::chrono::milliseconds kProgressInterval{ 250 };

        // Called with the lock held: the worker applies it.
        void want( MemoryHold hold )
        {
            wanted_ = hold;
            changed();
        }

        // Called with the lock held: a want or a backstop moved, so the worker looks again.
        void changed()
        {
            ++changes_;
            wake_.notify_all();
        }

        void work( std::stop_token stop )
        {
            std::unique_lock lock( mutex_ );

            while ( !stop.stop_requested() )
            {
                if ( release_at_ && std::chrono::steady_clock::now() >= *release_at_ )
                {
                    release_at_.reset();
                    wanted_ = MemoryHold::None;
                }

                if ( device_ && ( wanted_ != applied_ || device_ != applied_device_ ) )
                {
                    const std::shared_ptr<Device> device = device_;
                    const std::shared_ptr<Device> previous = applied_device_;
                    const MemoryHold hold = wanted_;

                    lock.unlock();

                    if ( previous && previous != device )
                    {
                        previous->setMemoryHold( MemoryHold::None );
                    }

                    apply( *device, hold );
                    lock.lock();

                    applied_ = hold;
                    applied_device_ = device;
                    applied_changed_.notify_all();

                    continue;
                }

                const std::uint64_t seen = changes_;
                const auto changed = [this, seen] { return changes_ != seen; };

                if ( release_at_ )
                {
                    wake_.wait_until( lock, stop, *release_at_, changed );
                }
                else
                {
                    wake_.wait( lock, stop, changed );
                }
            }
        }

        // Without the lock: a lock can take seconds. Its progress is the count of pages still out, read again on a
        // thread of its own, since the device locks in one call.
        void apply( Device& device, MemoryHold hold )
        {
            if ( hold == MemoryHold::None )
            {
                device.setMemoryHold( MemoryHold::None );
                Mila::Logging::Logger::info( "Mila's memory let go" );

                return;
            }

            const auto began = std::chrono::steady_clock::now();
            const std::size_t total = device.getMemoryHold() == MemoryHold::None ? device.getPagedOutBytes() : 0;

            if ( total == 0 )
            {
                device.setMemoryHold( hold );
            }
            else
            {
                progress_( 0, total );

                {
                    std::jthread reporter( [this, &device, total]( std::stop_token stop )
                        {
                            std::mutex pause;
                            std::condition_variable_any tick;
                            std::unique_lock wait( pause );

                            while ( !tick.wait_for( wait, stop, kProgressInterval, [] { return false; } ) && !stop.stop_requested() )
                            {
                                progress_( total - std::min( device.getPagedOutBytes(), total ), total );
                            }
                        } );

                    device.setMemoryHold( hold );
                }

                progress_( total, total );
            }

            Mila::Logging::Logger::info( std::format( "Mila's memory locked in {:.0f} ms, {} bytes of it brought back first",
                std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - began ).count(), total ) );
        }

        Progress progress_;
        std::mutex mutex_;
        std::condition_variable_any wake_;
        std::condition_variable_any applied_changed_;

        // What the window and turns want, and what the worker last made so; changes_ counts every want.
        std::shared_ptr<Device> device_;
        MemoryHold wanted_{ MemoryHold::None };
        std::shared_ptr<Device> applied_device_;
        MemoryHold applied_{ MemoryHold::None };
        std::uint64_t changes_{ 0 };
        std::optional<std::chrono::steady_clock::time_point> release_at_;

        // The window starts in front.
        bool active_{ true };

        // Last, so it stops before what it reads is destroyed.
        std::jthread worker_;
    };
}
