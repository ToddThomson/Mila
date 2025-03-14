module;  
#include <miniz.h>  
#include <vector>  
#include <string>  
#include <memory>  
#include <stdexcept>  
#include <type_traits>  
#include <sstream>  

export module Dnn.Module;  

import Dnn.Tensor;
import Dnn.TensorTraits;

import Compute.ComputeDevice;  
import Compute.CpuDevice;  
import Compute.CudaDevice;  

import Compute.ComputeResource;  
import Compute.CpuComputeResource;  
import Compute.CudaComputeResource;  

import Compute.MemoryResource;  
import Compute.CpuMemoryResource;  
import Compute.CudaMemoryResource;  

namespace Mila::Dnn  
{  
   /**  
   * @brief Abstract base class for all modules in the Mila DNN framework.  
   *  
   * @tparam TInput Data type of the input tensor elements.  
   * @tparam TCompute Data type of the compute tensor elements.  
   * @tparam TDevice Device type, either CpuDevice or CudaDevice.  
   */  
   export  
   template<typename TInput, typename TCompute = TInput, typename TDevice = Compute::CpuDevice>  
       requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>  
   class Module {  
   public:  
       using MR = TDevice::MR;  

       virtual ~Module() = default;  

       /**  
       * @brief Forward pass of the module.  
       *  
       * @param input Input tensor.  
       * @param output Output tensor.  
       */  
       virtual void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output ) = 0;  

       /**  
       * @brief Backward pass of the module.  
       *  
       * @param gradient Gradient tensor.  
       * @return Tensor<TCompute, MR> Gradient with respect to the input.  
       */  
       virtual Tensor<TCompute, MR> backward( const Tensor<TInput, MR>& gradient ) {  
           // Default to no op  
           return {};  
       }  

       /**  
       * @brief Set the training mode of the module.  
       *  
       * @param training True if the module is in training mode, false otherwise.  
       */  
       void setTrainingMode( bool training ) {  
           is_training_ = training;  
       }  

       /**  
       * @brief Check if the module is in training mode.  
       *  
       * @return true If the module is in training mode.  
       * @return false Otherwise.  
       */  
       bool isTraining() const {  
           return is_training_;  
       }  

       /**  
       * @brief Get the number of parameters in the module.  
       *  
       * @return size_t Number of parameters.  
       */  
       virtual size_t parameterCount() const = 0;  

       virtual const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getParameterTensors() const = 0;

       virtual const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getStateTensors() const = 0;

       /**  
       * @brief Get the name of the module.  
       *  
       * @return std::string Name of the module.  
       */  
       std::string getName() const {  
           return name_;  
       }  

       /**  
       * @brief Set the name of the module.  
       *  
       * @param name The name to set. Must not be empty and cannot contain a dot ('.').  
       * @throws std::invalid_argument If the name is empty or contains a dot.  
       */  
       void setName( const std::string& name ) {  
          if (name.empty() ) {  
              throw std::invalid_argument("Name must not be empty and cannot contain a dot ('.').");  
          }  
          name_ = name;  
       }  

       void setTraining( bool is_training ) {  
           is_training_ = is_training;  
       }  

       /**  
       * @brief Print the module information.  
       */  
       //virtual void print() const = 0;  

       /**  
       * @brief Save the module state to a file.  
       *  
       * @param zip The zip archive to save the state to.  
       */  
       virtual void save( mz_zip_archive& zip ) const = 0;  

       /**  
       * @brief Load the module state from a file.  
       *  
       * @param zip The zip archive to load the state from.  
       */  
       virtual void load( mz_zip_archive& zip ) = 0;  

       /**  
       * @brief Register a child module.  
       *  
       * @param name The name of the child module.  
       * @param module The child module to register.  
       * @throws std::runtime_error If a module with the same name is already registered.  
       */  
       void addModule( std::shared_ptr<Module<TInput, TCompute, TDevice>> module ) {  
           sub_modules_.emplace_back( module );  
       }  

       const std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>>& getSubModules() const {  
           return sub_modules_;  
       }  

       /**  
       * @brief Convert the module to a string representation.  
       *  
       * @return std::string String representation of the module.  
       */  
       virtual std::string toString() const = 0;  

       /**  
       * @brief Overload the << operator to print the module information.  
       *  
       * @param os Output stream.  
       * @param module Module to print.  
       * @return std::ostream& Output stream.  
       */  
       friend std::ostream& operator<<( std::ostream& os, const Module& module ) {  
           os << module.toString();  
           return os;  
       }  

   private:  
       std::string name_; ///< The name of the module. Cannot be empty and cannot contain a dot ('.').  
       bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.  

       std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>> sub_modules_; ///< Child modules.  
   };  
}