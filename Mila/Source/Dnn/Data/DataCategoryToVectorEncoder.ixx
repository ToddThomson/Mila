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
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>

export module Data.CategoryToVectorEncoder;

namespace Mila::Dnn::Data
{
    /// <summary>
    /// A category to vector (One-hot) encoder
    /// </summary>
    export template <typename TElement>
    class CategoryToVectorEncoder
    {
    public:

        /// <summary>
        /// Class constructor
        /// </summary>
        /// <param name="k">Size of vector</param>
        /// <param name="value">Numeric value to use for marking category</param>
        CategoryToVectorEncoder( size_t k, TElement value )
            : k_( k ), value_( value )
        {
            k_vector_ = std::vector<TElement>( k_, {} );
        }

        /// <summary>
        /// Encodes 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        std::vector<TElement> Convert( const std::vector<int>& input )
        {
            std::vector<TElement> output;
            output.reserve( k_ * input.size() );

            for (int e : input)
            {
                k_vector_[e - 1] = value_;
                std::copy( k_vector_.begin(), k_vector_.end(), std::back_inserter( output ) );
                k_vector_[e - 1] = {};
            }

            return output;
        }

    private:

        size_t k_ = 0;
        TElement value_;
        std::vector<TElement> k_vector_;
    };
}