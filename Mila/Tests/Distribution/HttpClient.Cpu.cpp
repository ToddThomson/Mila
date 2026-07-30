/**
 * @file HttpClient.Cpu.cpp
 * @brief Redirect resolution -- the pure part of the HTTP client, tested offline.
 *
 * These exist because the omission cost a live debugging round. Phase 1 shipped with no
 * tests on the argument that the client's contract was about live behaviour; the very first
 * real request then failed on relative-Location handling, which is pure string work and
 * needed no server at all.
 *
 * CPU only, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <string>

import Mila;

namespace Mila::Tests::Distribution
{
    using namespace Mila::Distribution;

    TEST( RedirectResolution, KeepsAnAbsoluteLocationUnchanged )
    {
        EXPECT_EQ(
            resolveRedirect( "https://huggingface.co/a/b/resolve/main/f.json",
                "https://cdn-lfs.hf.co/repos/xyz/f.json?token=abc" ),
            "https://cdn-lfs.hf.co/repos/xyz/f.json?token=abc" );
    }

    TEST( RedirectResolution, ResolvesTheRootRelativeFormHuggingFaceActuallySends )
    {
        // Verbatim shape of the 307 that broke the first live request: HuggingFace answers
        // a manifest fetch with a path, not a URL.
        const std::string base =
            "https://huggingface.co/mila-llm/gemma-4-12b-it/resolve/main/mila.json";
        const std::string location =
            "/api/resolve-cache/models/mila-llm/gemma-4-12b-it/2d88819/mila.json?etag=%22abc%22";

        EXPECT_EQ( resolveRedirect( base, location ),
            "https://huggingface.co"
            "/api/resolve-cache/models/mila-llm/gemma-4-12b-it/2d88819/mila.json?etag=%22abc%22" );
    }

    TEST( RedirectResolution, ResolvesAProtocolRelativeLocation )
    {
        EXPECT_EQ(
            resolveRedirect( "https://huggingface.co/a/b", "//cdn.example.com/x/y" ),
            "https://cdn.example.com/x/y" );
    }

    TEST( RedirectResolution, ResolvesAPathRelativeLocationAgainstTheCurrentDirectory )
    {
        EXPECT_EQ(
            resolveRedirect( "https://example.com/a/b/c.json", "d.json" ),
            "https://example.com/a/b/d.json" );

        // A query on the base must not leak into the resolved path.
        EXPECT_EQ(
            resolveRedirect( "https://example.com/a/b/c.json?x=1", "d.json" ),
            "https://example.com/a/b/d.json" );
    }

    TEST( RedirectResolution, HandlesABaseWithNoPath )
    {
        EXPECT_EQ( resolveRedirect( "https://example.com", "/x" ), "https://example.com/x" );
    }

    TEST( RedirectResolution, PreservesTheSchemeAcrossHosts )
    {
        // The host changes here, which is what drops the authorization header. The scheme
        // must survive so the next hop is still TLS.
        const std::string resolved =
            resolveRedirect( "https://huggingface.co/a", "//cdn-lfs.hf.co/b" );

        EXPECT_TRUE( resolved.starts_with( "https://" ) );
    }
}
