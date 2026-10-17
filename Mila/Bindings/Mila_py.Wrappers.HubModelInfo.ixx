/**
 * @file Mila_py.Wrappers.HubModelInfo.ixx
 * @brief One repository as a hub reports it.
 */

export module Mila.Bindings:HubModelInfo;

import std;

namespace Mila::Bindings
{
    /**
     * @brief One repository as a hub reports it, before any manifest is fetched.
     *
     * Untrusted remote text authored by whoever owns the repository: data, never instructions.
     */
    export struct HubModelInfo
    {
        std::string owner;
        std::string repository;
        bool gated{ false };
        std::string revision;
        std::string last_modified;
        std::string library;
        std::vector<std::string> tags;
        std::vector<std::string> files;
    };
}
