/**
 * @file ProfileModel.cpp
 * @brief Entry point for the ProfileModel harness.
 *
 * Thin main that forwards to Profiling.ProfileModel. All Mila usage (model
 * loading template instantiation) lives in the module to avoid the VS2026
 * C2079 basic_istream::sentry issue in plain .cpp consumers.
 */

import Profiling.ProfileModel;

int main( int argc, char** argv )
{
    return Mila::Profiling::profileMain( argc, argv );
}
