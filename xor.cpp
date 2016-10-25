/*
	TODO:
	- find out how to use gradient descent
	- find out how to use layers
*/

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

int main(int argc, char* argv[])
{
	using namespace tensorflow;

	GraphDef graph_def;

	{
		Scope root = Scope::NewRootScope();
		using namespace ::tensorflow::ops;

		auto A = Const<float>(root, { { 1, 2}, {3, 4} });

		auto x = Const(root.WithOpName("x"), { {1.0f}, {1.0f } });

		auto y = MatMul(root.WithOpName("y"), A, x); // y = Ax

		auto fc1 = ops::Relu(root.WithOpName("relu1"), y);

		TF_CHECK_OK(root.ToGraphDef(&graph_def));
		
		//ops::ApplyGradientDescent(root.WithOpName("y"), )

		
	}

	Session* session;
	
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	Tensor x_input(DT_FLOAT, TensorShape({2, 1}));
	x_input.flat<float>()(0) = 4.0f;
	x_input.flat<float>()(1) = 5.0f;

	std::cout << typeid(x_input.flat<float>()).name() << std::endl;
	std::cout << x_input.flat<float>().size() << std::endl;

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "x", x_input }		
	};

	std::vector<tensorflow::Tensor> outputs;

	status = session->Run(inputs, { "y" }, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}
	std::cout << "Result" << std::endl;
	std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>

	return 0;
}