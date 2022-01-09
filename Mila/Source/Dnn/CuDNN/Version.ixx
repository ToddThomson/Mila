/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

module;
#include <string>
#include <sstream>
#include <iostream>

export module CuDnn.Version;

namespace Mila::Dnn::CuDNN
{
    export struct cudnnVersion
    {
    public:

        cudnnVersion( int major, int minor, int patch )
            : major_( major ), minor_( minor ), patch_( patch )
        {
        }

        std::string ToString() const
        {
            std::stringstream ss;

            ss << std::to_string( major_ ) 
                << "." << std::to_string( minor_ )
                << "." << std::to_string( patch_ );

            return ss.str();
        }

        int getMajor() { return major_; };
        int getMinor() { return minor_; };
        int getPatch() { return patch_; };

    private:

        int major_;
        int minor_;
        int patch_;
    };
}