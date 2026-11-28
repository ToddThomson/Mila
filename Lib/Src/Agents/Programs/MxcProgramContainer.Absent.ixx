/**
 * @file MxcProgramContainer.Absent.ixx
 * @brief Agents.MxcProgramContainer where this build has no container: opening one always says why there is none.
 *
 * One of two candidate files for Agents.MxcProgramContainer; CMake compiles this one where no platform half is built,
 * which is every platform but Windows until macOS's and Linux's land (ContainedPrograms.md section 11, phase 4).
 */

export module Agents.MxcProgramContainer;

import std;
import Agents.ContainedRun;
import Agents.ProgramContainer;
import Agents.ProgramResult;

namespace Mila::Agents
{
    /// The container this build does not have; a grant given none refuses to run a program.
    export class MxcProgramContainer final : public ProgramContainer
    {
    public:

        /// @return Always why there is no container.
        static std::expected<std::shared_ptr<const MxcProgramContainer>, std::string> open( const std::filesystem::path& )
        {
            return std::unexpected( std::string( "Mila cannot run programs on this platform: its MXC container is built only for Windows" ) );
        }

        ProgramResult run( const ContainedRun&, std::stop_token ) const override
        {
            throw std::runtime_error( "this build of Mila runs no programs" );
        }

    private:

        MxcProgramContainer() = default;
    };
}
