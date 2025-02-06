module;
#include <vector>
#include <string>
#include <memory>

export module Dnn.Module;

import Dnn.Tensor;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

namespace Mila::Dnn
{
    export 
    template<typename T, typename MR> requires std::is_same_v<MR, Compute::CpuMemoryResource> || std::is_same_v<MR, Compute::DeviceMemoryResource>
    class Module {
    public:
        virtual ~Module() = default;

        virtual std::shared_ptr<Tensor<T, MR>> forward( const std::shared_ptr<Tensor<T,MR>> input ) = 0;

        virtual Tensor<T,MR> backward( const Tensor<T,MR>& gradient ) {
            // Default to no op
            return {};
        }

		void setTrainingMode( bool training ) {
			is_training_ = training;
		}

        bool isTraining() const {
            return is_training_;
        }

        virtual size_t parameters() const = 0;

        virtual std::string name() const = 0;

        virtual void print() const = 0;

    private:
        bool is_training_{ false };
    };
}