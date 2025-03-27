module;
#include <iostream>

export module Utils.DefaultLogger;

import Utils.Logger;

namespace Mila::Utils
{
    export class DefaultLogger : public Logger {
    public:
        void log( const std::string& message, int level = 0 ) override {
            if ( level >= log_level_ ) {
                std::cout << "[LOG] " << message << std::endl;
            }
        }

        void set_log_level( int level ) { log_level_ = level; }

    private:
        int log_level_ = 0;
    };
}