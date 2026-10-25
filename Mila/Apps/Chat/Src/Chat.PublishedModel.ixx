/**
 * @file Chat.PublishedModel.ixx
 * @brief One model the publisher offers, as far as its listing and its manifest say.
 */

export module Chat.PublishedModel;

import std;

namespace Mila::ChatApp
{
    /**
     * @brief One model the publisher offers, before it is installed.
     *
     * Names and licences are authored by whoever owns the repository: data to show, never to interpret.
     */
    export struct PublishedModel
    {
        /// The repository name, which is the name the store will hold it under.
        std::string name;

        /// False when the manifest could not be read; the fields below it are then empty.
        bool manifest_read{ false };

        /// What the install downloads: the manifest's files, not the repository's size.
        std::uint64_t download_bytes{ 0 };

        /// From the manifest, which is what governs once installed, rather than the listing's tag.
        std::string license;

        /// The weight format the manifest declares.
        std::string quantization;

        /// An instruct model of a family this build runs, in a format it reads.
        bool supported{ false };

        /// Its terms must be accepted on the hub before its files can be fetched.
        bool gated{ false };

        bool installed{ false };
    };
}
