#pragma once

namespace Mila::Dnn
{
    export template<typename T>
    class TensorBufferBase {
    public:
        virtual ~TensorBufferBase() = default;
        
        virtual size_t size() const = 0;
        
        virtual const T* data() const = 0;
        
        virtual T* data() = 0;
        
        virtual void fill( const T& value ) = 0;
        
        virtual void resize( size_t size ) = 0;
    };
}