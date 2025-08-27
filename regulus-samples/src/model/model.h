#include <maccel/maccel.h>

#include <vector>

using namespace std;
using namespace mobilint;

class NPUModel {
public:
    NPUModel(std::string modelPath);
    ~NPUModel();
    std::vector<std::vector<float>> operator()(std::unique_ptr<float[]> image);

    void release();

private:
    void buildModel(std::string modelPath);

    StatusCode mSc;
    unique_ptr<Accelerator> mAcc = Accelerator::create(mSc);
    unique_ptr<Model> mModel;
};
