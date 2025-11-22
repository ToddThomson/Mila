/**
 * @file Application.ixx
 * @brief Main application class for MilaViewer.
 */

module;
#include <Windows.h>
#include <memory>
#include <string>

export module MilaViewer.Application;

import Mila;
import MilaViewer.MainWindow;


namespace MilaViewer
{
    /**
     * @brief Main application singleton managing app lifecycle.
     */
    export class Application
    {
    public:
        static Application& instance()
        {
            static Application app;
            return app;
        }

        Application( const Application& ) = delete;
        Application& operator=( const Application& ) = delete;

        /**
         * @brief Initialize the application.
         */
        bool initialize( HINSTANCE hInstance )
        {
            hInstance_ = hInstance;

            // Initialize Mila framework
            if (!Mila::initialize())
            {
                MessageBoxW( nullptr, L"Failed to initialize Mila framework", L"Error", MB_OK | MB_ICONERROR );
                return false;
            }

            // Initialize COM for modern Windows features
            HRESULT hr = CoInitializeEx( nullptr, COINIT_APARTMENTTHREADED );
            if (FAILED( hr ))
            {
                MessageBoxW( nullptr, L"Failed to initialize COM", L"Error", MB_OK | MB_ICONERROR );
                return false;
            }

            // Initialize Common Controls
            INITCOMMONCONTROLSEX icex{};
            icex.dwSize = sizeof( INITCOMMONCONTROLSEX );
            icex.dwICC = ICC_WIN95_CLASSES | ICC_TREEVIEW_CLASSES | ICC_TAB_CLASSES;
            InitCommonControlsEx( &icex );

            // Create main window
            mainWindow_ = std::make_unique<MainWindow>();
            if (!mainWindow_->create( hInstance_ ))
            {
                MessageBoxW( nullptr, L"Failed to create main window", L"Error", MB_OK | MB_ICONERROR );
                return false;
            }

            return true;
        }

        /**
         * @brief Run the application message loop.
         */
        int run()
        {
            if (!mainWindow_)
                return -1;

            mainWindow_->show();

            MSG msg{};
            while (GetMessageW( &msg, nullptr, 0, 0 ))
            {
                TranslateMessage( &msg );
                DispatchMessageW( &msg );
            }

            return static_cast<int>(msg.wParam);
        }

        /**
         * @brief Shutdown the application.
         */
        void shutdown()
        {
            mainWindow_.reset();
            Mila::shutdown();
            CoUninitialize();
        }

        HINSTANCE getInstance() const
        {
            return hInstance_;
        }
        MainWindow* getMainWindow() const
        {
            return mainWindow_.get();
        }

    private:
        Application() = default;
        ~Application() = default;

        HINSTANCE hInstance_{ nullptr };
        std::unique_ptr<MainWindow> mainWindow_;
    };
}