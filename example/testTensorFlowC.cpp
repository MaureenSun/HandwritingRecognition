#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/standard_ops.h>

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    // Initialize TensorFlow session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << "Error creating TensorFlow session: " << status.ToString() << std::endl;
        return 1;
    }

    // Define the computation graph
    Scope root = Scope::NewRootScope();
    auto A = Const(root, {1.0, 2.0, 3.0, 4.0}, {2, 2});
    auto B = Const(root, {5.0, 6.0, 7.0, 8.0}, {2, 2});
    auto matmul = MatMul(root, A, B, MatMul::TransposeB(true));
    auto output = Identity(root, matmul);

    // Run the session
    std::vector<Tensor> outputs;
    status = session->Run({output}, &outputs);
    if (!status.ok()) {
        std::cerr << "Error running TensorFlow session: " << status.ToString() << std::endl;
        return 1;
    }

    // Output the result
    auto outputTensor = outputs[0].tensor<float, 2>();
    std::cout << "Result:" << std::endl << outputTensor << std::endl;

    // Close the session
    session->Close();
    delete session;

    return 0;
}
