module;
#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>

export module Serialization.ModelArchive;

import Dnn.Tensor;

namespace Mila::Dnn::Serialization
{
	using namespace Mila::Dnn;

	export class ModelArchive
	{
		public:
			ModelArchive(const std::string& directory_path, bool saving = false)
				: directory_path_( directory_path ), saving_( saving ) {}

			ModelArchive() = default;

		private:
			std::string directory_path_;
			bool saving_ = false;
			//nlohmann::json root_;
	};
}