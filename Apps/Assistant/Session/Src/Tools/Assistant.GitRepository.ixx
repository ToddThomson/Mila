/**
 * @file Assistant.GitRepository.ixx
 * @brief What Mila Assistant reads of a folder's git repository from its files, running no git: whether it has a
 *        commit yet, so a plan that started one ends by offering the first (Todd, 2026-11-27).
 */

export module Assistant.GitRepository;

import std;

namespace Mila::Assistant
{
    namespace
    {
        std::string firstLine( const std::filesystem::path& file )
        {
            std::ifstream in( file, std::ios::binary );
            std::string line;
            std::getline( in, line );

            while ( !line.empty() && ( line.back() == '\r' || line.back() == ' ' ) )
            {
                line.pop_back();
            }

            return line;
        }
    }

    /**
     * @brief Whether the folder holds a git repository whose branch has no commit: `.git/HEAD` names a branch that
     *        neither `.git/refs/heads` nor `packed-refs` holds. False for no repository, a detached head, or anything
     *        it cannot read, so a doubtful folder is never offered a "first" commit.
     */
    export bool repositoryWithoutCommits( const std::filesystem::path& folder )
    {
        std::error_code failed;
        const std::filesystem::path repository = folder / ".git";

        if ( !std::filesystem::is_directory( repository, failed ) )
        {
            return false;
        }

        const std::string head = firstLine( repository / "HEAD" );
        constexpr std::string_view kReference = "ref: ";

        if ( !head.starts_with( kReference ) )
        {
            return false;
        }

        const std::string branch = head.substr( kReference.size() );

        if ( branch.empty() || std::filesystem::exists( repository / branch, failed ) )
        {
            return false;
        }

        std::ifstream packed( repository / "packed-refs", std::ios::binary );

        for ( std::string line; std::getline( packed, line ); )
        {
            if ( line.ends_with( "\r" ) )
            {
                line.pop_back();
            }

            if ( line.ends_with( " " + branch ) )
            {
                return false;
            }
        }

        return true;
    }
}
