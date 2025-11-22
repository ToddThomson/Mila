/**
 * @file OutputView.ixx
 * @brief Tabbed output window for logs, validation results, etc.
 */

module;
#include <Windows.h>
#include <CommCtrl.h>
#include <string>
#include <vector>

export module MilaViewer.OutputView;

namespace MilaViewer
{
    /**
     * @brief Tabbed output pane for displaying logs and messages.
     */
    export class OutputView
    {
    public:
        OutputView() = default;
        ~OutputView()
        {
            if (hwndTab_ && IsWindow( hwndTab_ ))
            {
                DestroyWindow( hwndTab_ );
            }
        }

        bool create( HWND parent, HINSTANCE hInstance )
        {
            parent_ = parent;
            hInstance_ = hInstance;

            // Create tab control
            hwndTab_ = CreateWindowExW(
                0,
                WC_TABCONTROLW,
                nullptr,
                WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
                0, 0, 0, 0,
                parent,
                nullptr,
                hInstance,
                nullptr
            );

            if (!hwndTab_)
            {
                return false;
            }

            // Add tabs
            addTab( L"Output" );
            addTab( L"Validation" );
            addTab( L"Messages" );

            // Create edit control for each tab
            for (size_t i = 0; i < 3; ++i)
            {
                HWND hwndEdit = CreateWindowExW(
                    WS_EX_CLIENTEDGE,
                    L"EDIT",
                    nullptr,
                    WS_CHILD | WS_VSCROLL | WS_HSCROLL |
                    ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | ES_AUTOHSCROLL,
                    0, 0, 0, 0,
                    parent,
                    nullptr,
                    hInstance,
                    nullptr
                );

                if (!hwndEdit)
                {
                    return false;
                }

                // Set font
                HFONT hFont = CreateFontW(
                    -12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                    DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                    CLEARTYPE_QUALITY, FF_DONTCARE, L"Consolas"
                );
                SendMessageW( hwndEdit, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE );

                editWindows_.push_back( hwndEdit );
            }

            // Show first tab
            showTab( 0 );

            return true;
        }

        void resize( int x, int y, int width, int height )
        {
            if (!hwndTab_)
                return;

            SetWindowPos( hwndTab_, nullptr, x, y, width, height, SWP_NOZORDER );

            // Get tab control display area
            RECT rcTab;
            GetClientRect( hwndTab_, &rcTab );
            TabCtrl_AdjustRect( hwndTab_, FALSE, &rcTab );

            // Position edit controls
            for (HWND hwnd : editWindows_)
            {
                SetWindowPos( hwnd, nullptr,
                    x + rcTab.left,
                    y + rcTab.top,
                    rcTab.right - rcTab.left,
                    rcTab.bottom - rcTab.top,
                    SWP_NOZORDER );
            }
        }

        void appendText( const std::wstring& text, int tabIndex = 0 )
        {
            if (tabIndex < 0 || tabIndex >= static_cast<int>( editWindows_.size() ))
                return;

            HWND hwnd = editWindows_[tabIndex];
            int len = GetWindowTextLengthW( hwnd );
            SendMessageW( hwnd, EM_SETSEL, len, len );
            SendMessageW( hwnd, EM_REPLACESEL, FALSE,
                reinterpret_cast<LPARAM>( text.c_str() ) );
        }

        void clear( int tabIndex = 0 )
        {
            if (tabIndex < 0 || tabIndex >= static_cast<int>( editWindows_.size() ))
                return;

            SetWindowTextW( editWindows_[tabIndex], L"" );
        }

    private:
        void addTab( const std::wstring& name )
        {
            TCITEMW tie{};
            tie.mask = TCIF_TEXT;
            tie.pszText = const_cast<LPWSTR>(name.c_str());

            int index = TabCtrl_GetItemCount( hwndTab_ );
            TabCtrl_InsertItem( hwndTab_, index, &tie );
        }

        void showTab( int index )
        {
            for (size_t i = 0; i < editWindows_.size(); ++i)
            {
                ShowWindow( editWindows_[i], i == index ? SW_SHOW : SW_HIDE );
            }
        }

        HWND parent_{ nullptr };
        HINSTANCE hInstance_{ nullptr };
        HWND hwndTab_{ nullptr };
        std::vector<HWND> editWindows_;
    };
}