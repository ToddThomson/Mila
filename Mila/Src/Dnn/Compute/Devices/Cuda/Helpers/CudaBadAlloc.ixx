module;
#include <string>
#include <exception>

export module Cuda.BadAlloc;

namespace Mila::Dnn::Compute
{
    export class CudaBadAlloc : public std::bad_alloc {
    private:
        std::string _msg;
    public:
        explicit CudaBadAlloc( const std::string& msg ) : _msg( msg ) {}
        
        const char* what() const noexcept override { return _msg.c_str(); }
    };
}