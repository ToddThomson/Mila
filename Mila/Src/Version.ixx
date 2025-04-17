/*
 * Copyright 2025 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the Mila end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

module;
#include <string>
#include <sstream>
#include <iostream>
#include "Version.h"

export module Mila.Version;

namespace Mila
{
    /// <summary>
    /// Semantic Version data.
    /// </summary>
    export struct Version
    {
    public:

        /// <summary>
        /// Semantic Version Constructor
        /// </summary>
        /// <param name="major">Major API version</param>
        /// <param name="minor">Minor version for functional changes</param>
        /// <param name="patch">Patch for bug fixes</param>
        /// <param name="prerelease_tag">Optional pre-release text</param>
        /// <param name="prerelease">Optional pre-release text</param>
        Version( int major, int minor, int patch, const std::string& prerelease_tag = "", int prerelease = 0 )
            : major_( major ), minor_( minor ), patch_( patch ), pre_release_tag_( prerelease_tag ), pre_release_( prerelease )
        {
        }

        std::string ToString() const
        {
            std::stringstream ss;

            ss << std::to_string( major_ )
                << "." << std::to_string( minor_ )
                << "." << std::to_string( patch_ );

            if (!pre_release_tag_.empty())
            {
                ss << "-" << pre_release_tag_ << "." << std::to_string( pre_release_ );
            }

            return ss.str();
        }

        /// <summary>
        /// Gets the major version number.
        /// </summary>
        /// <returns>Major version</returns>
        int getMajor() { return major_; };

        /// <summary>
        /// Gets the minor version number.
        /// </summary>
        /// <returns>Minor version</returns>
        int getMinor() { return minor_; };

        /// <summary>
        /// Gets the patch version number.
        /// </summary>
        /// <returns>Patch version</returns>
        int getPatch() { return patch_; };

        /// <summary>
        /// Gets the pre-release version.
        /// </summary>
        /// <returns>Optional pre-release version</returns>
        std::string getPreRelease() { return pre_release_tag_ + "." + std::to_string(pre_release_); };

    private:

        int major_;
        int minor_;
        int patch_;
        std::string pre_release_tag_;
		int pre_release_;
    };
}