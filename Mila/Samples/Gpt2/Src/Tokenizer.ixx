module;
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <format>
#include <stdexcept>
#include <cctype>

export module Gpt2.Tokenizer;

namespace Mila::Dnn::Gpt2
{
	/// @brief Size of the GPT-2 tokenizer header.
	constexpr int Gpt2TokenizerHeaderSize = 256;
	/// @brief Magic number for the GPT-2 tokenizer.
	constexpr int Gpt2TokenizerMagicNumber = 20240328;

	/// @class Tokenizer
	/// @brief A class to handle GPT-2 tokenization.
	export class Tokenizer {
	public:
		/// @brief Constructs a Tokenizer object and initializes it with the given file.
		/// @param filename The name of the file containing the tokenizer data.
		Tokenizer(const std::string& filename)
			: vocab_size_(0), eot_token_(-1)
		{
			init(filename);
		}

		/// @brief Decodes a token ID to its corresponding string.
		/// @param token_id The ID of the token to decode.
		/// @return The decoded string corresponding to the token ID.
		const char* decode(uint32_t token_id) const {
			if (token_id < vocab_size_) {
				return token_table_[token_id].data();
			}
			else {
				throw std::runtime_error(std::format("Invalid token id: {}", token_id));
			}
		}

		/// @brief Gets the end-of-text token ID.
		/// @return The end-of-text token ID.
		int get_eot_token() const {
			return eot_token_;
		}

	private:
		/// @brief Initializes the tokenizer with the given file.
		/// @param filename The name of the file containing the tokenizer data.
		void init(const std::string& filename) {
			std::ifstream token_file(filename, std::ios::in | std::ifstream::binary);
			if (!token_file.is_open()) {
				throw std::runtime_error(std::format("Failed to open tokenizer file: {}", filename));
			}

			std::array<uint32_t, Gpt2TokenizerHeaderSize> tokenizer_header;
			token_file.read(reinterpret_cast<char*>(tokenizer_header.data()), Gpt2TokenizerHeaderSize * sizeof(uint32_t));

			// Check the magic number
			if (tokenizer_header[0] != Gpt2TokenizerMagicNumber) {
				throw std::runtime_error("Invalid magic number in tokenizer file");
			}

			int version = tokenizer_header[1];
			vocab_size_ = tokenizer_header[2];

			if (version == 1) {
				if (vocab_size_ != 50257) {
					throw std::runtime_error("Invalid vocab size in tokenizer file");
				}
				eot_token_ = 50256;
			}
			else if (version == 2) {
				eot_token_ = tokenizer_header[3];
			}
			else {
				std::cerr << "Tokenizer model file " << filename << " has bad version: " << version << "\n";
				throw std::runtime_error("Invalid version in tokenizer file");
			}

			unsigned char length;
			token_table_.resize(vocab_size_);

			for (uint32_t i = 0; i < vocab_size_; i++) {
				token_file.read(reinterpret_cast<char*>(&length), sizeof(unsigned char));
				std::vector<char> token_bytes(length + 1);
				token_file.read(token_bytes.data(), length * sizeof(char));
				token_bytes[length] = '\0';
				token_table_[i] = std::move(token_bytes);
			}
		}

		/// @brief Safely prints a string to the console.
		/// @param piece The string to print.
		void safe_printf(const char* piece) const {
			if (piece == nullptr || piece[0] == '\0') {
				return;
			}
			if (piece[1] == '\0') {
				unsigned char byte_val = piece[0];
				if (!(std::isprint(byte_val) || std::isspace(byte_val))) {
					return;
				}
			}
			std::cout << piece;
		}

		uint32_t vocab_size_; ///< The size of the vocabulary.
		std::vector<std::vector<char>> token_table_; ///< The table of tokens.
		int eot_token_; ///< The end-of-text token ID.
	};
}