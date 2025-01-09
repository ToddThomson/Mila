/**
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

module;
#include <cuda_runtime.h>

export module Cuda.Stream;

import Cuda.UniqueHandle;

namespace Mila::Dnn::Cuda
{
	export class CudaStream : public CudaUniqueHandle<cudaStream_t, CudaStream>
	{
	public:

		CudaUniqueHandle<cudaStream_t, CudaStream>::CudaUniqueHandle;
		CudaUniqueHandle<cudaStream_t, CudaStream>::operator=;

		/// <summary>
		/// Creates a stream on specified device (or current device, if device_id < 0)
		/// </summary>
		/// <param name="non_blocking"></param>
		/// <param name="device_id"></param>
		/// <returns></returns>
		static CudaStream Create( bool non_blocking, int device_id = -1 )
		{
			cudaStream_t stream;
			int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
			//DeviceGuard dg(device_id);
			cudaStreamCreateWithFlags( &stream, flags );

			return CudaStream( stream );
		}

		/// <summary>
		/// Creates a stream with given priority on specified device
		/// (or current device, if device_id < 0)
		/// </summary>
		/// <param name="non_blocking"></param>
		/// <param name="priority"></param>
		/// <param name="device_id"></param>
		/// <returns></returns>
		static CudaStream CreateWithPriority( bool non_blocking, int priority, int device_id = -1 )
		{
			cudaStream_t stream;
			int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
			//DeviceGuard dg(device_id);
			cudaStreamCreateWithPriority( &stream, flags, priority );

			return CudaStream( stream );
		}

		/// <summary>
		/// Calls cudaStreamDestroy on the handle.
		/// </summary>
		/// <param name="stream"></param>
		static void DestroyHandle( cudaStream_t stream )
		{
			cudaStreamDestroy( stream );
		}

	private:

		cudaStream_t stream_id_;
	};
}