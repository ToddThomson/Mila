/**
 * @file ModelTreeView.ixx
 * @brief Tree view for displaying network module hierarchy.
 */

module;
#include <Windows.h>
#include <CommCtrl.h>
#include <string>
#include <memory>

export module MilaViewer.ModelTreeView;

import Mila;

namespace MilaViewer
{
    /**
     * @brief Tree control displaying Mila network module hierarchy.
     */
    export class ModelTreeView
    {
    public:
        ModelTreeView() = default;
        ~ModelTreeView()
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
                WC_TREEVIEWW,
                nullptr,
                WS_CHILD | WS_VISIBLE | WS_BORDER |
                TVS_HASBUTTONS | TVS_HASLINES | TVS_LINESATROOT | TVS_SHOWSELALWAYS,
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

            // Set up tree view appearance
            SendMessageW( hwnd_, TVM_SETBKCOLOR, 0, RGB( 255, 255, 255 ) );
            SendMessageW( hwnd_, TVM_SETTEXTCOLOR, 0, RGB( 0, 0, 0 ) );

            // Populate with sample data
            populateSampleModel();

            return true;
        }

        void resize( int x, int y, int width, int height )
        {
            if (hwnd_)
            {
                SetWindowPos( hwnd_, nullptr, x, y, width, height, SWP_NOZORDER );
            }
        }

        void refresh()
        {
            if (hwnd_)
            {
                TreeView_DeleteAllItems( hwnd_ );
                populateSampleModel();
            }
        }

        /**
         * @brief Load a Mila network into the tree view.
         */
        template<Mila::Dnn::Compute::DeviceType TDeviceType>
        void loadNetwork( const Mila::Dnn::Network<TDeviceType>* network )
        {
            if (!hwnd_ || !network)
                return;

            TreeView_DeleteAllItems( hwnd_ );

            // Add root node for network
            TVINSERTSTRUCTW tvis{};
            tvis.hParent = TVI_ROOT;
            tvis.hInsertAfter = TVI_LAST;
            tvis.item.mask = TVIF_TEXT | TVIF_STATE;
            tvis.item.stateMask = TVIS_EXPANDED;
            tvis.item.state = TVIS_EXPANDED;

            std::wstring networkName = L"Network: " + std::wstring( network->getName().begin(),
                network->getName().end() );
            tvis.item.pszText = const_cast<LPWSTR>(networkName.c_str());

            HTREEITEM hRoot = TreeView_InsertItem( hwnd_, &tvis );

            // Add child modules
            const auto& modules = network->getModules();
            for (const auto& module : modules)
            {
                addModuleNode( hRoot, module.get() );
            }
        }

    private:
        void populateSampleModel()
        {
            // Create sample model structure
            TVINSERTSTRUCTW tvis{};
            tvis.hParent = TVI_ROOT;
            tvis.hInsertAfter = TVI_LAST;
            tvis.item.mask = TVIF_TEXT | TVIF_STATE;
            tvis.item.stateMask = TVIS_EXPANDED;
            tvis.item.state = TVIS_EXPANDED;
            tvis.item.pszText = const_cast<LPWSTR>(L"MnistClassifier");

            HTREEITEM hRoot = TreeView_InsertItem( hwnd_, &tvis );

            // Add child nodes
            addNode( hRoot, L"Linear (fc1)", L"784 ? 128" );
            addNode( hRoot, L"GELU (gelu1)", L"activation" );
            addNode( hRoot, L"Linear (fc2)", L"128 ? 64" );
            addNode( hRoot, L"GELU (gelu2)", L"activation" );
            addNode( hRoot, L"Linear (output)", L"64 ? 10" );
        }

        HTREEITEM addNode( HTREEITEM parent, const std::wstring& name, const std::wstring& info = L"" )
        {
            std::wstring displayText = name;
            if (!info.empty())
            {
                displayText += L" [" + info + L"]";
            }

            TVINSERTSTRUCTW tvis{};
            tvis.hParent = parent;
            tvis.hInsertAfter = TVI_LAST;
            tvis.item.mask = TVIF_TEXT;
            tvis.item.pszText = const_cast<LPWSTR>(displayText.c_str());

            return TreeView_InsertItem( hwnd_, &tvis );
        }

        template<Mila::Dnn::Compute::DeviceType TDeviceType>
        void addModuleNode( HTREEITEM parent, Mila::Dnn::Module<TDeviceType>* module )
        {
            if (!module)
                return;

            std::wstring moduleName( module->getName().begin(), module->getName().end() );
            std::wstring paramCount = L"params: " + std::to_wstring( module->parameterCount() );

            addNode( parent, moduleName, paramCount );
        }

        HWND hwnd_{ nullptr };
    };
}