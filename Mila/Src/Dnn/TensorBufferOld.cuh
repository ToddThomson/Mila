#pragma once

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/fill.h>
//#include "TensorBufferBase.h"
//
//namespace Mila::Dnn
//{
//    template<typename T, template<typename> class VectorType = thrust::device_vector>
//    class TensorBuffer : public TensorBufferBase<T> {
//    public:
//        TensorBuffer( size_t buffer_size );
//
//        size_t size() const override;
//        const T* data() const override;
//        T* data() override;
//        void fill( const T& value ) override;
//        void resize( size_t size ) override;
//
//    private:
//        VectorType<T> buffer_;
//    };
//}
