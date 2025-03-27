module;
#include <string>

export module Utils.Logger;

namespace Mila::Utils
{
    export class Logger {
    public:
        virtual void log( const std::string& message, int level = 0 ) = 0;
        virtual ~Logger() = default;
    };
}