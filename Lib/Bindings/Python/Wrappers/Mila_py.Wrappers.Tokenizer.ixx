/**
 * @file Mila_py.Wrappers.Tokenizer.ixx
 * @brief A model's tokenizer, behind a std-only handle.
 */

export module Mila.Bindings:Tokenizer;

import std;

namespace Mila::Bindings
{
    export class Tokenizer
    {
    public:
        static std::shared_ptr<Tokenizer> loadLlama32( const std::string& path );
        static std::shared_ptr<Tokenizer> loadGemma( const std::string& path );
        static std::shared_ptr<Tokenizer> loadQwen( const std::string& path );

        /**
         * @brief The tokenizer of an installed model, by store name.
         *
         * Which loader to use is a property of the weights, not of the caller, so the
         * record decides it. That is what removes the pairing a consumer previously had to
         * keep correct by hand: a tokenizer path and a weights path that had to describe
         * the same model, with nothing checking that they did.
         *
         * @throws std::runtime_error if no such model is installed, its files are missing,
         *         or its architecture has no tokenizer in this binding.
         */
        static std::shared_ptr<Tokenizer> fromStore( const std::string& name );

        std::vector<std::int32_t> encode( const std::string& text );
        std::string decode( const std::vector<std::int32_t>& ids );
        std::string tokenToString( std::int32_t token_id ) const;
        bool isValidToken( std::int32_t token_id ) const;
        std::int64_t vocabSize() const;
        std::optional<std::int32_t> bosTokenId() const;
        std::optional<std::int32_t> eosTokenId() const;
        std::optional<std::int32_t> padTokenId() const;

        ~Tokenizer();

    private:
        struct Impl;
        explicit Tokenizer( std::unique_ptr<Impl> impl );

        std::unique_ptr<Impl> impl_;
    };
}
