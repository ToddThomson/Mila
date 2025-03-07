module;
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <type_traits>
#include <cuda_runtime.h>

export module Dnn.Graph;

//import Dnn.Module;
//import Dnn.Tensor;
//
//namespace Mila::Dnn
//{
//    /*export
//    template<typename TInput, typename TCompute = TInput, typename MR = Compute::CpuMemoryResource>
//        requires ValidTensorTypes<TInput, TCompute> && (std::is_same_v<MR, Compute::CpuMemoryResource> || std::is_same_v<MR, Compute::DeviceMemoryResource>)
//    class Model : public Module<TInput, TCompute, MR> {
//    public:*/
//
//    //    // Register a node and return its output tensor
//    //    std::shared_ptr<Tensor<TInput,MR>> registerNode( std::shared_ptr<Node<T>> node ) {
//    //        nodes.push_back( node );
//    //        return tensor_map[ node->output_id ];  // Return associated tensor
//    //    }
//
//    //    // Topological sorting to determine execution order
//    //    void build() {
//    //        std::unordered_set<std::shared_ptr<Node<T>>> visited;
//    //        execution_order.clear();
//
//    //        for ( auto& node : nodes ) {
//    //            if ( visited.find( node ) == visited.end() ) {
//    //                dfs( node, visited );
//    //            }
//    //        }
//    //    }
//
//    //    void dfs( std::shared_ptr<Node<T>> node, std::unordered_set<std::shared_ptr<Node<T>>>& visited ) {
//    //        visited.insert( node );
//    //        for ( const auto& input_id : node->input_ids ) {
//    //            if ( tensor_map.count( input_id ) ) {
//    //                auto input_node = tensor_map[ input_id ];  // Retrieve corresponding tensor
//    //                if ( input_node && visited.find( node ) == visited.end() ) {
//    //                    dfs( node, visited );
//    //                }
//    //            }
//    //        }
//    //        execution_order.push_back( node );
//    //    }
//
//    //    // Execute forward pass
//    //    void forward() {
//    //        for ( auto& node : execution_order ) {
//    //            node->forward( tensor_map );
//    //        }
//    //    }
//
//    //    // Execute backward pass
//    //    void backward( const std::shared_ptr<Tensor<T>>& loss ) {
//    //        loss->grad.assign( loss->value.size(), 1.0 );
//    //        for ( auto it = execution_order.rbegin(); it != execution_order.rend(); ++it ) {
//    //            (*it)->backward( tensor_map );
//    //        }
//    //    }
//
//    //private:
//    //    std::unordered_map<std::string, std::shared_ptr<Tensor<T>>> tensor_map;
//    //    std::vector<std::shared_ptr<Node<T>>> nodes;
//    //    std::vector<std::shared_ptr<Node<T>>> execution_order;
//    };
//
//
//    //export
//    //    template <typename T>
//    //class CudaGraph {
//    //    std::vector<std::shared_ptr<Node<T>>> execution_order;
//    //    std::unordered_map<std::string, std::shared_ptr<Tensor<T>>> tensor_map;
//    //    cudaGraph_t cuda_graph;
//    //    cudaGraphExec_t cuda_graph_exec;
//    //    bool cuda_graph_initialized = false;
//
//    //public:
//    //    std::shared_ptr<Tensor<T>> tensor( size_t size ) {
//    //        auto t = std::make_shared<Tensor<T>>( size );
//    //        tensor_map[ t->getID() ] = t;
//    //        return t;
//    //    }
//
//    //    // Create a Linear layer
//    //    std::shared_ptr<Tensor<T>> linear( std::shared_ptr<Tensor<T>> input,
//    //        std::shared_ptr<Tensor<T>> weight,
//    //        std::shared_ptr<Tensor<T>> bias ) {
//    //        auto node = std::make_shared<Linear<T>>( input, weight, bias );
//    //        return registerNode( node );
//    //    }
//
//    //    std::shared_ptr<Tensor<T>> gelu( std::shared_ptr<Tensor<T>> input ) {
//    //        auto node = std::make_shared<GELU<T>>( input );
//    //        return registerNode( node );
//    //    }
//
//    //    void buildCUDA() {
//    //        cudaGraphCreate( &cuda_graph, 0 );
//    //        cudaStream_t stream;
//    //        cudaStreamCreate( &stream );
//
//    //        for ( auto& node : execution_order ) {
//    //            node->recordCUDA( cuda_graph, stream );
//    //        }
//
//    //        cudaGraphInstantiate( &cuda_graph_exec, cuda_graph, nullptr, nullptr, 0 );
//    //        cuda_graph_initialized = true;
//    //    }
//
//    //    void executeCUDA() {
//    //        if ( !cuda_graph_initialized ) {
//    //            buildCUDA();
//    //        }
//    //        cudaGraphLaunch( cuda_graph_exec, 0 );
//    //        cudaStreamSynchronize( 0 );
//    //    }
//    //};
//}