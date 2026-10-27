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
     * A `--model` that does not resolve exits to the shell, whose remedies are `mila-store` commands. Inside a
     * session a refusal names no command, because the console and the Mila app do not share any.
     */
    export enum class RefusalAudience
    {
        Session,
        Shell
    };
}
