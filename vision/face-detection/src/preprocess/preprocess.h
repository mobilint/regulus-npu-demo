#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

class PreProcessor {
public:
    PreProcessor();
    PreProcessor(int imh, int imw, bool is_ssd = false, bool auto_padding = false,
                 int stride = 32);
    void set(int imh, int imw, bool is_ssd = false, bool auto_padding = false,
             int stride = 32);
    std::unique_ptr<float[]> operator()(cv::Mat image);

private:
    cv::Mat letter_box(cv::Mat image, cv::Size im_shape, bool auto_padding = false,
                       int stride = 32);

    int m_imh = 640;
    int m_imw = 640;
    bool m_is_ssd = false;
    bool m_auto_padding = false;
    int m_stride = 32;
};
