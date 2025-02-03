//module;
//#include <math.h>
//#include <iostream>
//
//export module Compute.Operations;
//
//import Compute.CudaMatMulOp;
//import Compute.CpuMatMulOp;
////import Compute.Cuda.LayerNormOp;
//import Compute.CpuLayerNormOp;
//
//namespace Mila::Dnn::Compute
//{
//	export class Operations {
//	public:
//		static Operations& instance() {
//			static Operations instance;
//			
//			// Lazy initialization of operations
//			if ( !is_initialized_ ) {
//				// CPU operations...
//				CpuMatMulOp<float>::registerOperation();
//
//				// CUDA operations...
//				CudaMatMulOp<float>::registerOperation();
//
//				is_initialized_ = true;
//			}
//
//			return instance;
//		}
//
//		Operations( const Operations& ) = delete;
//		Operations operator=( const Operations& ) = delete;
//
//	private:
//		Operations() = default;
//		static inline bool is_initialized_ = false;
//	};
//}