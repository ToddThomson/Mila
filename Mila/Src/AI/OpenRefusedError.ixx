/**
 * @file OpenRefusedError.ixx
 * @brief A refused open, thrown by an open that checked the name and found it would not open.
 */

export module AI.OpenRefusedError;

import std;
export import AI.OpenRefusal;

namespace Mila::AI
{
    /**
     * @brief Thrown with the refusal the same name's check returns, so a program that catches it and one that checks
     *        first read one answer.
     */
    export class OpenRefusedError : public std::runtime_error
    {
    public:

        explicit OpenRefusedError( OpenRefusal refusal )
            : std::runtime_error( refusal.detail ), refusal_( std::move( refusal ) )
        {
        }

        [[nodiscard]] const OpenRefusal& refusal() const noexcept
        {
            return refusal_;
        }

    private:

        OpenRefusal refusal_;
    };
}
