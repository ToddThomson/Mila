
#include "Mila.h"

#include <string>
#include <sstream>
#include <ostream>

#include "SampleRnnApp.h"
#include "SampleRnnModel.h"

using namespace std;
using namespace Mila::Dnn;
using namespace Mila::Dnn::CuDNN;
using namespace RnnSample;

int main()
{
	std::cout << "CuDNN Version: "
		<< GetVersion().ToString() << endl;

	auto model = SampleRnnModel();

	cout << "NN App initialized."; // << app.ToString() << endl;

	return 0;
}


