/**
 * @file DocumentView.ixx
 * @brief Document area for displaying model details and visualizations.
 */

module;
#include <Windows.h>
#include <string>

export module MilaViewer.DocumentView;

namespace MilaViewer
{
    /**
     * @brief Document view pane for displaying model information.
     */
    export class DocumentView
    {
    public:
        DocumentView() = default;
        ~DocumentView()
        {
            if (hwnd_ && IsWindow( hwnd_ ))
            {
                DestroyWindow( hwnd_ );
            }
        }

        bool create( HWND parent, HINSTANCE hInstance )
        {
            hwnd_ = CreateWindowExW(
                WS_EX_CLIENTEDGE,
                L"EDIT",
                nullptr,
                WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL |
                ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | ES_AUTOHSCROLL,
                0, 0, 0, 0,
                parent,
                nullptr,
                hInstance,
                nullptr
            );

            if (!hwnd_)
            {
                return false;
            }

            // Set font
            HFONT hFont = CreateFontW(
                -14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                CLEARTYPE_QUALITY, FF_DONTCARE, L"Consolas"
            );
            SendMessageW( hwnd_, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE );

            // Set default content
            std::wstring defaultText =
                L"Mila Network Viewer\n"
                L"==================\n\n"
                L"Open a model file to view its architecture and details.\n\n"
                L"Supported operations:\n"
                L"  • View network module hierarchy\n"
                L"  • Inspect module parameters and configurations\n"
                L"  • Validate model structure\n"
                L"  • Export architecture documentation\n";

            SetWindowTextW( hwnd_, defaultText.c_str() );

            return true;
        }

        void resize( int x, int y, int width, int height )
        {
            if (hwnd_)
            {
                SetWindowPos( hwnd_, nullptr, x, y, width, height, SWP_NOZORDER );
            }
        }

        void setText( const std::wstring& text )
        {
            if (hwnd_)
            {
                SetWindowTextW( hwnd_, text.c_str() );
            }
        }

        void appendText( const std::wstring& text )
        {
            if (hwnd_)
            {
                int len = GetWindowTextLengthW( hwnd_ );
                SendMessageW( hwnd_, EM_SETSEL, len, len );
                SendMessageW( hwnd_, EM_REPLACESEL, FALSE,
                    reinterpret_cast<LPARAM>(text.c_str()) );
            }
        }

    private:
        HWND hwnd_{ nullptr };
    };
}