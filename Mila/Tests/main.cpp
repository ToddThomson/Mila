#include <gtest/gtest.h>

import Mila;

class FailureThresholdListener : public testing::EmptyTestEventListener {
private:
    int max_failures_;
    int failure_count_;

public:
    explicit FailureThresholdListener( int max_failures )
        : max_failures_( max_failures ), failure_count_( 0 ) {}

    void OnTestPartResult( const testing::TestPartResult& test_part_result ) override {
        if ( test_part_result.failed() ) {
            ++failure_count_;

            if ( failure_count_ >= max_failures_ ) {
                std::cerr << "\n*** TEST RUN ABORTED AFTER " << max_failures_
                    << " FAILURES ***\n" << std::endl;
                // Force exit after max failures reached
                std::exit( 1 );
            }
        }
    }
};

int main( int argc, char** argv ) {
    Mila::initialize();

    ::testing::InitGoogleTest( &argc, argv );

    // testing::GTEST_FLAG(break_on_failure) = true;

    const int MAX_FAILURES = 10;
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append( new FailureThresholdListener( MAX_FAILURES ) );

    return RUN_ALL_TESTS();
}