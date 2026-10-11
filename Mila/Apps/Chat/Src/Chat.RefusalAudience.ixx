/**
 * @file Chat.RefusalAudience.ixx
 * @brief Where the user stands when a refusal is read, which decides the commands it names.
 */

export module Chat.RefusalAudience;

namespace Mila::ChatApp
{
    /**
     * @brief Where the user stands when a refusal is read, which decides the commands it names.
     *
     * A `--model` that does not resolve exits to the shell, where no slash command exists.
     */
    export enum class RefusalAudience
    {
        Session,
        Shell
    };
}
