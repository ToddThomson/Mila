#ifndef MILA_DNN_CUDNN_VERSION_H_
#define MILA_DNN_CUDNN_VERSION_H_

namespace Mila::Dnn::CuDNN 
{
    struct cudnnVersion
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
#endif