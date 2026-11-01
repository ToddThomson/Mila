/**
 * @file FolderAccess.ixx
 * @brief What a grant lets a tool do in one folder: read it, or read and write it.
 */

export module Milo.FolderAccess;

import std;

namespace Milo
{
    /**
     * @brief A granted folder's access (ToolCalling.md 3.3).
     */
    export enum class FolderAccess
    {
        /// Files in the folder can be opened and read, and its folders listed.
        Read,

        /// Files can also be written, replaced and created, and folders made.
        ReadWrite,
    };
}
