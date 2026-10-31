/**
 * @file MxcProgramContainer.Absent.ixx
 * @brief AI.MxcProgramContainer where this build has no container: opening one always says why there is none.
 *
 * One of two candidate files for AI.MxcProgramContainer; CMake compiles this one unless the build is for Windows and
 * MILA_ENABLE_MXC is on. Nothing of MXC is built or shipped with it (ContainedPrograms.md 4.11).
 */

export module AI.MxcProgramContainer;

import std;
import AI.ContainedRun;
import AI.ProgramContainer;
import AI.ProgramResult;

namespace Mila::AI
{
    /// The container this build does not have; a grant given none refuses to run a program.
    export class MxcProgramContainer final : public ProgramContainer
    {
    public:

        /// @return Always why there is no container.
        static std::expected<std::shared_ptr<const MxcProgramContainer>, std::string> open( const std::filesystem::path& )
        {
            return std::unexpected( std::string( "this build of Mila runs no programs: it was built without MXC (MILA_ENABLE_MXC)" ) );
        }

        ProgramResult run( const ContainedRun&, std::stop_token ) const override
        {
            throw std::runtime_error( "this build of Mila runs no programs" );
        }

    private:

        MxcProgramContainer() = default;
    };
}
