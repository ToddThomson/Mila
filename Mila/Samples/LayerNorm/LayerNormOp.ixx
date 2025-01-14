module;
#include <math.h>
#include <iostream>

export module App.Ops.LayerNorm;

import Mila;

using namespace Mila::Dnn;

namespace App::Ops
{
    export template<typename T>
    class LayerNormOp : public Module<T> {
    public:
        
        LayerNormOp( std::string name, size_t batch_size, size_t sequence_length, size_t channels ) 
            : name_(name), B_( batch_size ), T_(sequence_length), C_( channels ) {
            init_tensors();
        }

        // --------------------------------------------------------------------
        // Properties..

        Tensor<T>& Weight() {
            return weight_;
        }

        Tensor<T>& Bias() {
            return bias_;
        }

        // --------------------------------------------------------------------
        // Module interface methods..

        Tensor<T> forward( const Tensor<T>& input ) override {
            Tensor<T> output( input.shape() );
            float eps = 1e-5f;
            
            for ( int b = 0; b < B_; b++ ) {
                for ( int t = 0; t < T_; t++ ) {
                    // seek to the input position inp[b,t,:]
                    // TJT: was float* x = inp + b * T * C_ + t * C_;
                    int input_offset = b * T_ * C_ + t * C_;
                    
                    // calculate the mean
                    float m = 0.0f;
                    for ( int i = 0; i < C_; i++ ) {
                        m += input[ input_offset + i ];
                    }
                    m = m / C_;
                    
                    // calculate the variance (without any bias correction)
                    float v = 0.0f;
                    for ( int i = 0; i < C_; i++ ) {
                        float xshift = input[ input_offset + i ] - m;
                        v += xshift * xshift;
                    }
                    v = v / C_;
                    
                    // calculate the rstd
                    float s = 1.0f / sqrtf( v + eps );
                    
                    // seek to the output position in out[b,t,:]
                    // TJT: was float* out_bt = out + b * T_ * C_ + t * C_;
                    int out_offset = b * T_ * C_ + t * C_;
                    
                    for ( int i = 0; i < C_; i++ ) {
                        float n = (s * (input[ input_offset + i ] - m)); // normalized output
                        float o = n * weight_[ i ] + bias_[ i ]; // scale and shift it
                        output[ out_offset + i ] = o; // write
                    }
                    
                    // cache the mean and rstd for the backward pass later
                    mean_[ b * T_ + t ] = m;
                    rstd_[ b * T_ + t ] = s;
                }
            }

            return output;
        }

        void backward( float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, float* mean, float* rstd, int B, int T, int C ) {
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    float* dout_bt = dout + b * T * C + t * C;
                    float* inp_bt = inp + b * T * C + t * C;
                    float* dinp_bt = dinp + b * T * C + t * C;
                    float mean_bt = mean[ b * T + t ];
                    float rstd_bt = rstd[ b * T + t ];

                    // first: two reduce operations
                    float dnorm_mean = 0.0f;
                    float dnorm_norm_mean = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        float norm_bti = (inp_bt[ i ] - mean_bt) * rstd_bt;
                        float dnorm_i = weight[ i ] * dout_bt[ i ];
                        dnorm_mean += dnorm_i;
                        dnorm_norm_mean += dnorm_i * norm_bti;
                    }
                    dnorm_mean = dnorm_mean / C;
                    dnorm_norm_mean = dnorm_norm_mean / C;

                    // now iterate again and accumulate all the gradients
                    for ( int i = 0; i < C; i++ ) {
                        float norm_bti = (inp_bt[ i ] - mean_bt) * rstd_bt;
                        float dnorm_i = weight[ i ] * dout_bt[ i ];
                        // gradient contribution to bias
                        dbias[ i ] += dout_bt[ i ];
                        // gradient contribution to weight
                        dweight[ i ] += norm_bti * dout_bt[ i ];
                        // gradient contribution to input
                        float dval = 0.0f;
                        dval += dnorm_i; // term 1
                        dval -= dnorm_mean; // term 2
                        dval -= norm_bti * dnorm_norm_mean; // term 3
                        dval *= rstd_bt; // final scale
                        dinp_bt[ i ] += dval;
                    }
                }
            }
        }
    
		size_t parameters() const override {
			return C_ * 2;
		}

		std::string name() const override {
            return name_;
		}

		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameters: " << parameters() << std::endl;
		}


    private:
		std::string name_{ "LayerNormOp" };
        size_t B_{ 0 };
        size_t T_{ 0 };
        size_t C_{ 0 };

        Tensor<T> weight_ = Tensor<T>( { C_ } );
        Tensor<T> bias_ = Tensor<T>( { C_ } );
        
        Tensor<T> mean_ = Tensor<T>( { B_ * T_ } );
        Tensor<T> rstd_ = Tensor<T>( { B_ * T_ } );
        //Tensor<T> y = Tensor<T>( { B_ * T_ * C_ } );

        void init_tensors() {
            xavier( weight_, C_, C_ );
        }
    };
}