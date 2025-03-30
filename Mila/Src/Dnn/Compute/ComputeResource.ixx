/**
 * @file ComputeResource.ixx
 * @brief Base class for compute resources used in the neural network framework.
 */

export module Compute.ComputeResource;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Abstract base class for compute resources.
	 *
	 * The ComputeResource class serves as a non-copyable, non-movable base class
	 * for device-specific compute resource implementations (CPU, CUDA, etc.).
	 * It provides a common interface for managing computational resources required
	 * for neural network operations.
	 *
	 * Derived classes like HostComputeResource and DeviceComputeResource typically
	 * extend this class to provide device-specific implementations that manage
	 * memory resources and other device capabilities.
	 *
	 * @note This class follows the RAII (Resource Acquisition Is Initialization) principle
	 *       for resource management but delegates the actual resource handling to derived classes.
	 *
	 * @see HostComputeResource
	 * @see DeviceComputeResource
	 */
	export class ComputeResource {
	public:
		/**
		 * @brief Default constructor.
		 *
		 * Creates an empty compute resource. Derived classes are responsible for
		 * initializing any specific resources they manage.
		 */
		ComputeResource() = default;

		/**
		 * @brief Copy constructor (deleted).
		 *
		 * ComputeResource instances cannot be copied because they typically
		 * manage unique system resources.
		 */
		ComputeResource( const ComputeResource& ) = delete;

		/**
		 * @brief Copy assignment operator (deleted).
		 *
		 * ComputeResource instances cannot be copied because they typically
		 * manage unique system resources.
		 */
		ComputeResource& operator=( const ComputeResource& ) = delete;

		/**
		 * @brief Move constructor (deleted).
		 *
		 * ComputeResource instances cannot be moved because they typically
		 * represent non-transferable system resources.
		 */
		ComputeResource( ComputeResource&& ) = delete;

		/**
		 * @brief Move assignment operator (deleted).
		 *
		 * ComputeResource instances cannot be moved because they typically
		 * represent non-transferable system resources.
		 */
		ComputeResource& operator=( ComputeResource&& ) = delete;

		/**
		 * @brief Virtual destructor.
		 *
		 * Allows derived classes to properly clean up their resources.
		 */
		virtual ~ComputeResource() = default;
	};
}
