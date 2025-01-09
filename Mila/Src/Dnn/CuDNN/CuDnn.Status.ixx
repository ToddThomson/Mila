/*
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
#include <cudnn.h>

export module CuDnn.Status;

namespace Mila::Dnn::CuDnn
{
	export template <typename T>
	struct cudnnResult
	{
	public:

		cudnnResult() = default;

		cudnnResult( cudnnStatus_t status_, T value )
			: status_( status_ ), value_( value )
		{
		}

		bool IsSuccess()
		{
			return (status_ == CUDNN_STATUS_SUCCESS);
		}

		cudnnStatus_t getStatus()
		{
			return status_;
		}

		T getValue()
		{
			return value_;
		}

	private:

		cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;

		T value_;
	};
}