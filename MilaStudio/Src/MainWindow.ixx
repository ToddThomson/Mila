/**
 * @file MainWindow.ixx
 * @brief Main window with 3-pane layout for model visualization.
 */

module;
#include <Windows.h>
#include <CommCtrl.h>
#include <string>
#include <memory>

export module MilaViewer.MainWindow;

import MilaViewer.ModelTreeView;
import MilaViewer.DocumentView;
import MilaViewer.OutputView;
import Mila;

namespace MilaViewer
{
    /**
     * @brief Main application window with toolbar, menu, and 3-pane layout.
     *
     * Layout:
     * +-------------------+
     * | Menu   | Toolbar  |
     * +------+------------+
     * | Tree | Document   |
     * |      +------------+
     * |      | Output     |
     * +------+------------+
     */
    export class MainWindow
    {
    public:
        MainWindow() = default;
        ~MainWindow()
        {
            if (hwnd_ && IsWindow( hwnd_ ))
            {
                DestroyWindow( hwnd_ );
            }
        }

        bool create( HINSTANCE hInstance )
        {
            hInstance_ = hInstance;

            // Register window class
            WNDCLASSEXW wcex{};
            wcex.cbSize = sizeof( WNDCLASSEXW );
            wcex.style = CS_HREDRAW | CS_VREDRAW;
            wcex.lpfnWndProc = WindowProc;
            wcex.hInstance = hInstance;
            wcex.hIcon = LoadIcon( nullptr, IDI_APPLICATION );
            wcex.hCursor = LoadCursor( nullptr, IDC_ARROW );
            wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
            wcex.lpszMenuName = nullptr;
            wcex.lpszClassName = L"MilaViewerMainWindow";
            wcex.hIconSm = LoadIcon( nullptr, IDI_APPLICATION );

            if (!RegisterClassExW( &wcex ))
            {
                return false;
            }

            // Create window
            hwnd_ = CreateWindowExW(
                WS_EX_APPWINDOW,
                L"MilaViewerMainWindow",
                L"Mila Network Viewer",
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT, CW_USEDEFAULT,
                1600, 1000,
                nullptr,
                nullptr,
                hInstance,
                this
            );

            if (!hwnd_)
            {
                return false;
            }

            // Create child components
            createMenu();
            createToolbar();
            createPanes();

            return true;
        }

        void show() const
        {
            ShowWindow( hwnd_, SW_SHOWMAXIMIZED );
            UpdateWindow( hwnd_ );
        }

        HWND getHandle() const
        {
            return hwnd_;
        }

    private:
        static LRESULT CALLBACK WindowProc( HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam )
        {
            MainWindow* window = nullptr;

            if (msg == WM_NCCREATE)
            {
                auto cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
                window = static_cast<MainWindow*>(cs->lpCreateParams);
                SetWindowLongPtrW( hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window) );
                window->hwnd_ = hwnd;
            }
            else
            {
                window = reinterpret_cast<MainWindow*>(GetWindowLongPtrW( hwnd, GWLP_USERDATA ));
            }

            if (window)
            {
                return window->handleMessage( msg, wParam, lParam );
            }

            return DefWindowProcW( hwnd, msg, wParam, lParam );
        }

        LRESULT handleMessage( UINT msg, WPARAM wParam, LPARAM lParam )
        {
            switch (msg)
            {
                case WM_CREATE:
                    return 0;

                case WM_SIZE:
                    onResize( LOWORD( lParam ), HIWORD( lParam ) );
                    return 0;

                case WM_COMMAND:
                    return handleCommand( LOWORD( wParam ) );

                case WM_NOTIFY:
                    return handleNotify( reinterpret_cast<LPNMHDR>(lParam) );

                case WM_DESTROY:
                    PostQuitMessage( 0 );
                    return 0;
            }

            return DefWindowProcW( hwnd_, msg, wParam, lParam );
        }

        void createMenu()
        {
            HMENU hMenuBar = CreateMenu();

            // File menu
            HMENU hFileMenu = CreatePopupMenu();
            AppendMenuW( hFileMenu, MF_STRING, IDM_FILE_OPEN, L"&Open Model...\tCtrl+O" );
            AppendMenuW( hFileMenu, MF_STRING, IDM_FILE_RECENT, L"Open &Recent" );
            AppendMenuW( hFileMenu, MF_SEPARATOR, 0, nullptr );
            AppendMenuW( hFileMenu, MF_STRING, IDM_FILE_CLOSE, L"&Close Model\tCtrl+W" );
            AppendMenuW( hFileMenu, MF_SEPARATOR, 0, nullptr );
            AppendMenuW( hFileMenu, MF_STRING, IDM_FILE_EXIT, L"E&xit\tAlt+F4" );
            AppendMenuW( hMenuBar, MF_POPUP, reinterpret_cast<UINT_PTR>(hFileMenu), L"&File" );

            // View menu
            HMENU hViewMenu = CreatePopupMenu();
            AppendMenuW( hViewMenu, MF_STRING, IDM_VIEW_REFRESH, L"&Refresh\tF5" );
            AppendMenuW( hViewMenu, MF_SEPARATOR, 0, nullptr );
            AppendMenuW( hViewMenu, MF_STRING | MF_CHECKED, IDM_VIEW_TREE, L"Model &Tree" );
            AppendMenuW( hViewMenu, MF_STRING | MF_CHECKED, IDM_VIEW_OUTPUT, L"&Output Window" );
            AppendMenuW( hMenuBar, MF_POPUP, reinterpret_cast<UINT_PTR>(hViewMenu), L"&View" );

            // Tools menu
            HMENU hToolsMenu = CreatePopupMenu();
            AppendMenuW( hToolsMenu, MF_STRING, IDM_TOOLS_VALIDATE, L"&Validate Model" );
            AppendMenuW( hToolsMenu, MF_STRING, IDM_TOOLS_EXPORT, L"&Export Architecture..." );
            AppendMenuW( hToolsMenu, MF_SEPARATOR, 0, nullptr );
            AppendMenuW( hToolsMenu, MF_STRING, IDM_TOOLS_OPTIONS, L"&Options..." );
            AppendMenuW( hMenuBar, MF_POPUP, reinterpret_cast<UINT_PTR>(hToolsMenu), L"&Tools" );

            // Help menu
            HMENU hHelpMenu = CreatePopupMenu();
            AppendMenuW( hHelpMenu, MF_STRING, IDM_HELP_DOCS, L"&Documentation" );
            AppendMenuW( hHelpMenu, MF_STRING, IDM_HELP_ABOUT, L"&About Mila Viewer..." );
            AppendMenuW( hMenuBar, MF_POPUP, reinterpret_cast<UINT_PTR>(hHelpMenu), L"&Help" );

            SetMenu( hwnd_, hMenuBar );
        }

        void createToolbar()
        {
            hwndToolbar_ = CreateWindowExW(
                0,
                TOOLBARCLASSNAMEW,
                nullptr,
                WS_CHILD | WS_VISIBLE | TBSTYLE_FLAT | TBSTYLE_TOOLTIPS,
                0, 0, 0, 0,
                hwnd_,
                reinterpret_cast<HMENU>(IDC_TOOLBAR),
                hInstance_,
                nullptr
            );

            SendMessageW( hwndToolbar_, TB_BUTTONSTRUCTSIZE, sizeof( TBBUTTON ), 0 );

            TBBUTTON buttons[] = {
                { STD_FILEOPEN, IDM_FILE_OPEN, TBSTATE_ENABLED, BTNS_BUTTON, {0}, 0, (INT_PTR)L"Open" },
                { 0, 0, TBSTATE_ENABLED, BTNS_SEP, {0}, 0, 0 },
                { STD_PROPERTIES, IDM_TOOLS_VALIDATE, TBSTATE_ENABLED, BTNS_BUTTON, {0}, 0, (INT_PTR)L"Validate" },
                { 0, 0, TBSTATE_ENABLED, BTNS_SEP, {0}, 0, 0 },
                { STD_HELP, IDM_HELP_DOCS, TBSTATE_ENABLED, BTNS_BUTTON, {0}, 0, (INT_PTR)L"Help" },
            };

            SendMessageW( hwndToolbar_, TB_ADDBUTTONSW, sizeof( buttons ) / sizeof( TBBUTTON ),
                reinterpret_cast<LPARAM>(buttons) );
        }

        void createPanes()
        {
            // Create left pane (model tree)
            treeView_ = std::make_unique<ModelTreeView>();
            treeView_->create( hwnd_, hInstance_ );

            // Create right-top pane (document view)
            documentView_ = std::make_unique<DocumentView>();
            documentView_->create( hwnd_, hInstance_ );

            // Create right-bottom pane (output view)
            outputView_ = std::make_unique<OutputView>();
            outputView_->create( hwnd_, hInstance_ );
        }

        void onResize( int width, int height )
        {
            // Resize toolbar
            if (hwndToolbar_)
            {
                SendMessageW( hwndToolbar_, TB_AUTOSIZE, 0, 0 );

                RECT rcToolbar;
                GetWindowRect( hwndToolbar_, &rcToolbar );
                toolbarHeight_ = rcToolbar.bottom - rcToolbar.top;
            }

            // Calculate pane dimensions
            const int treeWidth = 300;
            const int outputHeight = 250;
            const int splitterSize = 5;

            int clientWidth = width;
            int clientHeight = height - toolbarHeight_;
            int rightWidth = clientWidth - treeWidth - splitterSize;
            int docHeight = clientHeight - outputHeight - splitterSize;

            // Position tree view (left)
            if (treeView_)
            {
                treeView_->resize( 0, toolbarHeight_, treeWidth, clientHeight );
            }

            // Position document view (right-top)
            if (documentView_)
            {
                documentView_->resize( treeWidth + splitterSize, toolbarHeight_,
                    rightWidth, docHeight );
            }

            // Position output view (right-bottom)
            if (outputView_)
            {
                outputView_->resize( treeWidth + splitterSize,
                    toolbarHeight_ + docHeight + splitterSize,
                    rightWidth, outputHeight );
            }
        }

        LRESULT handleCommand( WORD commandId )
        {
            switch (commandId)
            {
                case IDM_FILE_OPEN:
                    onFileOpen();
                    return 0;

                case IDM_FILE_EXIT:
                    PostMessageW( hwnd_, WM_CLOSE, 0, 0 );
                    return 0;

                case IDM_VIEW_REFRESH:
                    onViewRefresh();
                    return 0;

                case IDM_TOOLS_VALIDATE:
                    onToolsValidate();
                    return 0;

                case IDM_HELP_ABOUT:
                    onHelpAbout();
                    return 0;
            }

            return 0;
        }

        LRESULT handleNotify( LPNMHDR pnmh )
        {
            // Handle notifications from child controls
            return 0;
        }

        void onFileOpen()
        {
            OPENFILENAMEW ofn{};
            wchar_t filename[MAX_PATH] = {};

            ofn.lStructSize = sizeof( OPENFILENAMEW );
            ofn.hwndOwner = hwnd_;
            ofn.lpstrFilter = L"Mila Model Files (*.mila)\0*.mila\0All Files (*.*)\0*.*\0";
            ofn.lpstrFile = filename;
            ofn.nMaxFile = MAX_PATH;
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

            if (GetOpenFileNameW( &ofn ))
            {
                loadModel( filename );
            }
        }

        void loadModel( const std::wstring& path )
        {
            // TODO: Load model using Mila API
            if (outputView_)
            {
                outputView_->appendText( L"Loading model: " + path + L"\n" );
            }
        }

        void onViewRefresh()
        {
            if (treeView_)
            {
                treeView_->refresh();
            }
        }

        void onToolsValidate()
        {
            if (outputView_)
            {
                outputView_->appendText( L"Validating model...\n" );
            }
        }

        void onHelpAbout()
        {
            std::wstring about = L"Mila Network Viewer\n\n";
            about += L"Version 1.0.0\n\n";
            about += L"A tool for visualizing and analyzing Mila neural network models.\n\n";
            about += L"Copyright © 2025";

            MessageBoxW( hwnd_, about.c_str(), L"About Mila Viewer", MB_OK | MB_ICONINFORMATION );
        }

        // Menu command IDs
        static constexpr WORD IDM_FILE_OPEN = 101;
        static constexpr WORD IDM_FILE_RECENT = 102;
        static constexpr WORD IDM_FILE_CLOSE = 103;
        static constexpr WORD IDM_FILE_EXIT = 104;
        static constexpr WORD IDM_VIEW_REFRESH = 201;
        static constexpr WORD IDM_VIEW_TREE = 202;
        static constexpr WORD IDM_VIEW_OUTPUT = 203;
        static constexpr WORD IDM_TOOLS_VALIDATE = 301;
        static constexpr WORD IDM_TOOLS_EXPORT = 302;
        static constexpr WORD IDM_TOOLS_OPTIONS = 303;
        static constexpr WORD IDM_HELP_DOCS = 401;
        static constexpr WORD IDM_HELP_ABOUT = 402;
        static constexpr WORD IDC_TOOLBAR = 500;

        HINSTANCE hInstance_{ nullptr };
        HWND hwnd_{ nullptr };
        HWND hwndToolbar_{ nullptr };
        int toolbarHeight_{ 0 };

        std::unique_ptr<ModelTreeView> treeView_;
        std::unique_ptr<DocumentView> documentView_;
        std::unique_ptr<OutputView> outputView_;
    };
}