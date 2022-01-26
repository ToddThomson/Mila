/*
 * Copyright 2022 Todd Thomson, Achilles Software.  All rights reserved.
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
#include <string_view>
#include <fmt/format.h>
#include <fmt/color.h>
#include <iostream>

export module Core.Logger;

namespace Mila::Core
{
    export enum LogLevel {
        None = 0,
        Error = 1,
        Warn = 2,
        Trace = 3,
        Info = 4
    };

    export class Logger {

    public:

        static void set_level( LogLevel level ) {
            log_level_ = level;
        }

        template <typename... T>
        static void log( LogLevel level, fmt::format_string<T...> fmt, T&&... args )
        {
            if ( level <= log_level_ ) {
                return;
            }

            std::cout << fmt::format( fmt, args...) << std::endl;
        }

        static LogLevel log_level_;
    };

    LogLevel Logger::log_level_ = LogLevel::None;
}