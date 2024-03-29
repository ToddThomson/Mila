#/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
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

// Original license reference

// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;
#include <algorithm>

export module Cuda.UniqueHandle;

namespace Mila::Dnn::Cuda
{
    export template <typename THandleType, typename TResourceType>
    class CudaUniqueHandle
    {
    public:

        using handle_t = THandleType;

        constexpr inline CudaUniqueHandle() : handle_( TResourceType::null_handle() )
        {
        }

        constexpr explicit CudaUniqueHandle( handle_t handle ) : handle_( handle )
        {
        }

        CudaUniqueHandle( const CudaUniqueHandle& ) = delete;

        CudaUniqueHandle& operator=( const CudaUniqueHandle& ) = delete;

        inline CudaUniqueHandle( CudaUniqueHandle&& other ) : handle_( other.handle_ )
        {
            other.handle_ = TResourceType::null_handle();
        }

        inline CudaUniqueHandle& operator=( CudaUniqueHandle&& other )
        {
            std::swap( handle_, other.handle_ );
            other.reset();
            return *this;
        }

        constexpr operator handle_t() const noexcept
        {
            return handle_;
        }

        inline void reset()
        {
            if ( !TResourceType::is_null_handle( handle_ ) )
            {
                TResourceType::DestroyHandle( handle_ );
                handle_ = TResourceType::null_handle();
            }
        }

        inline void reset( handle_t handle )
        {
            if ( handle != handle_ )
            {
                reset();
                handle_ = handle;
            }
        }

        inline handle_t release() noexcept
        {
            handle_t old = handle_;
            handle_ = TResourceType::null_handle();
            
            return old;
        }

        constexpr explicit operator bool() const noexcept
        {
            return !TResourceType::is_null_handle( handle_ );
        }

        static constexpr handle_t null_handle() noexcept
        {
            return {};
        }

        static constexpr bool is_null_handle( const handle_t& handle ) noexcept
        {
            return handle == TResourceType::null_handle();
        }

    protected:

        inline ~CudaUniqueHandle()
        {
            reset();
        }

        handle_t handle_;
    };
}