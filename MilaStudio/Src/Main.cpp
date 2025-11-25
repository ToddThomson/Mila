/**
 * @file Main.cpp
 * @brief Entry point for MilaViewer application.
 */

#include <Windows.h>

import MilaViewer.Application;

int WINAPI wWinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine,
    _In_ int nShowCmd )
{
    (void)hPrevInstance;
    (void)lpCmdLine;
    (void)nShowCmd;

    auto& app = MilaViewer::Application::instance();

    if (!app.initialize( hInstance ))
    {
        return -1;
    }

    int result = app.run();

    app.shutdown();

    return result;
}