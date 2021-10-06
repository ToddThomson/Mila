// MilaApp.cpp : Defines the entry point for the application.

#include "MilaApp.h"
#include "Mila.h"

using namespace Gerb;
using namespace std;

int main()
{
    int r = Gerb::myAddFunc(10, 10);

	cout << "Hello CMake." << endl;

    cout << " Adding 10 + 10 = " << r << endl;

	return 0;
}
