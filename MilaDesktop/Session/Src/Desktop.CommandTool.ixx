/**
 * @file Desktop.CommandTool.ixx
 * @brief Mila Desktop's command tool: one Windows PowerShell command run in the project folder through the call's
 *        grant, held to it by the grant's container (ContainedPrograms.md section 5).
 */

export module Desktop.CommandTool;

import std;
import nlohmann.json;

import Mila;
import Milo;

namespace Mila::Desktop
{
    using Milo::ProgramRequest;
    using Milo::ProgramResult;
    using Milo::ToolGrant;
    using Milo::ToolNeeds;

    /**
     * @brief Runs a command the model wrote and gives it the exit code and what the command printed.
     *
     * The command reaches the network only when its call asks for it, and then only as far as the grant allows:
     * the grant the tool is given holds the network, and a call that does not ask has it narrowed away (decision
     * 11). Every call is asked, so the person sees which.
     */
    export class CommandTool
    {
    public:

        static constexpr std::string_view kName = "run_command";

        /// What the person is shown the tool touches; a call without `network` is given less (run).
        static ToolNeeds needs() noexcept
        {
            ToolNeeds needs;
            needs.reads_files = true;
            needs.writes_files = true;
            needs.uses_network = true;
            needs.runs_programs = true;

            return needs;
        }

        /// Whether a call's arguments ask for the network: only `"network": true` does.
        [[nodiscard]] static bool asksForNetwork( std::string_view arguments_text )
        {
            const auto arguments = nlohmann::json::parse( arguments_text, nullptr, false );

            return arguments.is_object() && arguments.contains( "network" ) && arguments[ "network" ].is_boolean()
                && arguments[ "network" ].get<bool>();
        }

        /// `command`, and optionally `time_seconds` and `network`.
        [[nodiscard]] static std::string run( const std::string& arguments_text, const ToolGrant& grant, std::stop_token stop )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string command = arguments.at( "command" ).get<std::string>();
            const bool network = asksForNetwork( arguments_text );
            int seconds = kDefaultSeconds;

            if ( arguments.contains( "time_seconds" ) && arguments[ "time_seconds" ].is_number_integer() )
                seconds = std::clamp( arguments[ "time_seconds" ].get<int>(), 1, kMostSeconds );

            if ( command.find_first_not_of( " \t\r\n" ) == std::string::npos )
                throw std::runtime_error( "the command is empty" );

            ToolNeeds given = needs();
            given.uses_network = network;

            // Written through the grant, so the script lies in the project as any file a tool writes there does.
            grant.openFileToWrite( kScriptPath, true ).replace( script( command ) );

            ProgramRequest request;
            request.command_line = powershellCommandLine();
            request.time = std::chrono::seconds( seconds );
            request.output_bytes = kMostOutputBytes;

            const ProgramResult result = grant.narrowedTo( given ).runProgram( request, stop );

            return describe( result, seconds );
        }

    private:

        static constexpr int kDefaultSeconds = 120;
        static constexpr int kMostSeconds = 600;
        static constexpr std::size_t kMostOutputBytes = 32u * 1024u;

        /// Relative to the project, where the run starts.
        static constexpr std::string_view kScriptPath = ".mila/command.ps1";

        /**
         * @brief The command as a script: output written as UTF-8, and an exit code that is the last program's
         *        when it failed, else 1 when the last command failed.
         *
         * The byte order mark makes Windows PowerShell 5.1 read the script as UTF-8, not the system's code page.
         */
        static std::string script( const std::string& command )
        {
            return "\xEF\xBB\xBF[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
                "$ProgressPreference = 'SilentlyContinue'\n"
                + command + "\n"
                "$succeeded = $?\n"
                "if ( $LASTEXITCODE ) { exit $LASTEXITCODE }\n"
                "if ( -not $succeeded ) { exit 1 }\n";
        }

        /**
         * @brief Windows PowerShell 5.1 running the script from its file, so no quoting of the model's text can
         *        change the command line, and errors come back as text (`-EncodedCommand` sends them as CLIXML).
         *
         * Execution policy is set for this process only, so `npm`, which PowerShell finds as `npm.ps1`, runs.
         */
        static std::string powershellCommandLine()
        {
            const char* system_root = std::getenv( "SystemRoot" );
            const std::string root = system_root != nullptr && *system_root != '\0' ? system_root : "C:\\Windows";

            return "\"" + root + "\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -NoLogo -NoProfile -NonInteractive "
                "-ExecutionPolicy Bypass -File .mila\\command.ps1";
        }

        static std::string describe( const ProgramResult& result, int seconds )
        {
            std::string text;

            if ( result.stopped )
                text = "The command was stopped before it finished.\n";
            else if ( result.timed_out )
                text = std::format( "The command ran out of its {} seconds and was ended; give a longer time_seconds if it needs one.\n", seconds );
            else
                text = std::format( "Exit code: {}\n", result.exit_code );

            appendStream( text, "Output", result.output, result.output_dropped );
            appendStream( text, "Errors", result.errors, result.errors_dropped );

            if ( !result.withheld.empty() )
            {
                text += "\nNot readable by the command, since each has another name outside the folder:\n";

                for ( const std::string& name : result.withheld )
                    text += "  " + name + "\n";
            }

            return text;
        }

        static void appendStream( std::string& text, std::string_view label, const std::string& stream, std::uint64_t dropped )
        {
            if ( stream.empty() && dropped == 0 )
                return;

            text += std::format( "\n{}:\n{}", label, stream );

            if ( !stream.empty() && stream.back() != '\n' )
                text += "\n";

            if ( dropped > 0 )
                text += std::format( "({} more bytes were not kept)\n", dropped );
        }
    };
}
