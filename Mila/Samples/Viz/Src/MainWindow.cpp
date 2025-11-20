// VisualizerWindow.cpp
// Build: cl /std:c++23 VisualizerWindow.cpp d2d1.lib windowscodecs.lib

#include <windows.h>
#include <d2d1.h>
#include <chrono>
#include <thread>
#include <vector>

#pragma comment(lib, "d2d1")

constexpr int WIDTH = 3840;
constexpr int HEIGHT = 2160;

// Global D2D Objects
ID2D1Factory* g_factory = nullptr;
ID2D1HwndRenderTarget* g_renderTarget = nullptr;
ID2D1Bitmap* g_bitmap = nullptr;

// CPU framebuffer (RGB24)
std::vector<uint8_t> g_framebuffer;

// Forward declare
void RenderFrame();

//-------------------------------------------------------------------------------------------------
// Create D2D render target and bitmap attached to the HWND
//-------------------------------------------------------------------------------------------------
HRESULT CreateGraphicsResources( HWND hwnd )
{
    HRESULT hr = S_OK;

    if (!g_renderTarget)
    {
        RECT rc;
        GetClientRect( hwnd, &rc );

        D2D1_SIZE_U size = D2D1::SizeU( rc.right, rc.bottom );

        hr = g_factory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(
                D2D1_RENDER_TARGET_TYPE_DEFAULT,
                D2D1::PixelFormat( DXGI_FORMAT_B8G8R8A8_UNORM,
                    D2D1_ALPHA_MODE_IGNORE ) // 24bit input+dummy alpha
            ),
            D2D1::HwndRenderTargetProperties( hwnd, size ),
            &g_renderTarget
        );

        if (FAILED( hr )) return hr;

        // Create D2D bitmap matching framebuffer
        D2D1_BITMAP_PROPERTIES props =
        {
            D2D1::PixelFormat(
                DXGI_FORMAT_B8G8R8A8_UNORM,
                D2D1_ALPHA_MODE_IGNORE
            ),
            96.0f,
            96.0f
        };

        hr = g_renderTarget->CreateBitmap(
            D2D1::SizeU( WIDTH, HEIGHT ),
            nullptr,
            WIDTH * 4,            // pitch (BGRA)
            props,
            &g_bitmap
        );
    }
    return hr;
}

//-------------------------------------------------------------------------------------------------
void DiscardGraphicsResources()
{
    if (g_bitmap) g_bitmap->Release(); g_bitmap = nullptr;
    if (g_renderTarget) g_renderTarget->Release(); g_renderTarget = nullptr;
}

//-------------------------------------------------------------------------------------------------
void OnPaint( HWND hwnd )
{
    if (FAILED( CreateGraphicsResources( hwnd ) )) return;

    PAINTSTRUCT ps;
    BeginPaint( hwnd, &ps );

    g_renderTarget->BeginDraw();

    // Convert RGB24 framebuffer ? BGRA32 buffer
    // (directly write into a temp scanline)
    static std::vector<uint8_t> bgra( WIDTH * HEIGHT * 4 );

    const uint8_t* src = g_framebuffer.data();
    uint8_t* dst = bgra.data();

    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
        dst[i * 4 + 0] = src[i * 3 + 2];  // B
        dst[i * 4 + 1] = src[i * 3 + 1];  // G
        dst[i * 4 + 2] = src[i * 3 + 0];  // R
        dst[i * 4 + 3] = 255;             // A (ignored)
    }

    // Copy into the Direct2D bitmap
    D2D1_RECT_U rect = { 0,0, WIDTH, HEIGHT };
    g_bitmap->CopyFromMemory( &rect, bgra.data(), WIDTH * 4 );

    // Draw the bitmap scaled to window size
    D2D1_SIZE_F rtSize = g_renderTarget->GetSize();
    g_renderTarget->DrawBitmap(
        g_bitmap,
        D2D1::RectF( 0, 0, rtSize.width, rtSize.height )
    );

    HRESULT hr = g_renderTarget->EndDraw();
    if (hr == D2DERR_RECREATE_TARGET)
    {
        DiscardGraphicsResources();
    }

    EndPaint( hwnd, &ps );
}

//-------------------------------------------------------------------------------------------------
void RenderFrameAndInvalidate( HWND hwnd )
{
    RenderFrame();      // Your renderer fills g_framebuffer
    InvalidateRect( hwnd, nullptr, FALSE );
}

//-------------------------------------------------------------------------------------------------
LRESULT CALLBACK WindowProc( HWND hwnd, UINT msg, WPARAM w, LPARAM l )
{
    switch (msg)
    {
        case WM_PAINT:
            OnPaint( hwnd );
            return 0;

        case WM_DESTROY:
            DiscardGraphicsResources();
            PostQuitMessage( 0 );
            return 0;
    }
    return DefWindowProc( hwnd, msg, w, l );
}

//-------------------------------------------------------------------------------------------------
// Your visualization fill
// Replace with your transformer drawing pipeline
//-------------------------------------------------------------------------------------------------
void RenderFrame()
{
    static float t = 0.0f;
    t += 0.02f;

    // Simple test pattern (moving rainbow)
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++)
        {
            float fx = (float)x / WIDTH;
            float fy = (float)y / HEIGHT;

            int r = int( (sin( fx * 10 + t ) * 0.5 + 0.5) * 255 );
            int g = int( (sin( fy * 10 + t ) * 0.5 + 0.5) * 255 );
            int b = int( (sin( (fx + fy) * 10 + t ) * 0.5 + 0.5) * 255 );

            size_t idx = (y * WIDTH + x) * 3;
            g_framebuffer[idx + 0] = (uint8_t)r;
            g_framebuffer[idx + 1] = (uint8_t)g;
            g_framebuffer[idx + 2] = (uint8_t)b;
        }
}

//-------------------------------------------------------------------------------------------------
int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE, LPSTR, int )
{
    g_framebuffer.resize( WIDTH * HEIGHT * 3 );

    // Create D2D factory
    D2D1_FACTORY_OPTIONS options = {};
    D2D1CreateFactory(
        D2D1_FACTORY_TYPE_SINGLE_THREADED,
        options,
        &g_factory
    );

    // Register window class
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"D2DVisualizerWindow";
    RegisterClass( &wc );

    // Create 4K window
    HWND hwnd = CreateWindowEx(
        0,
        wc.lpszClassName,
        L"Transformer Visualizer (Direct2D)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        WIDTH / 2, HEIGHT / 2,   // window scaled down; D2D scales framebuffer
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    ShowWindow( hwnd, SW_SHOWDEFAULT );

    // 60 Hz timer
    using Clock = std::chrono::high_resolution_clock;
    auto next = Clock::now();

    MSG msg = {};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage( &msg, nullptr, 0, 0, PM_REMOVE ))
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        else
        {
            next += std::chrono::milliseconds( 16 ); // ~60Hz
            RenderFrameAndInvalidate( hwnd );
            std::this_thread::sleep_until( next );
        }
    }

    if (g_factory) g_factory->Release();
    return 0;
}
