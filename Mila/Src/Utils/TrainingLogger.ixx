module;
#include <fstream>
#include <iostream>

export module Utils.StepLogger;

namespace Mila::Utils
{
    export class StepLogger {
    public:
        StepLogger( std::string filename, int flush_after_steps = 20 )
        : flush_every_( flush_after_steps ) {
			if (filename.empty() ) {
				std::cerr << "No log file specified, logging to stdout" << std::endl;
			}
			else {
				logfile_ = std::ofstream( filename );
				if ( !logfile_.is_open() ) {
					std::cerr << std::format( "Failed to open log file {}, logging to stdout", filename ) << std::endl;
				}
			}
        }
		
        ~StepLogger() {
			if ( !logfile_.fail() ) {
				logfile_.close();
			}
		}

        void log_step( int step, std::string formatted_msg ) {
            if ( !logfile_.fail() ) {
                logfile_ << formatted_msg;
            }
            else {
				std::cout << formatted_msg;
            }

			if ( step % flush_every_ == 0 ) {
				logfile_.flush();
			}

        }

    private:

        std::ofstream logfile_;
        int flush_every_{ 20 }; // every how many steps to flush the log
    };
}