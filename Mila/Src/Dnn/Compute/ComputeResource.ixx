export module Compute.ComputeResource;

namespace Mila::Dnn::Compute
{
	export class ComputeResource {
	public:
		ComputeResource() = default;
		ComputeResource( const ComputeResource& ) = delete;
		ComputeResource& operator=( const ComputeResource& ) = delete;
		ComputeResource( ComputeResource&& ) = delete;
		ComputeResource& operator=( ComputeResource&& ) = delete;
		virtual ~ComputeResource() = default;
	};
}