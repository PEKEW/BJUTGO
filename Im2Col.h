

#ifndef IM2COL_H_INCLUDED
#define IM2COL_H_INCLUDED

#include <cassert>
#include <vector>
#include <algorithm>

template <unsigned long filter_size>
void im2col(const int channels,
            const std::vector<float>& input,
            std::vector<float>& output) {
    constexpr unsigned int height = BOARD_SIZE;
    constexpr unsigned int width = BOARD_SIZE;

    constexpr int pad = (filter_size / 2);
    constexpr unsigned int output_h = height + 2 * pad - filter_size  + 1;
    constexpr unsigned int output_w = width + 2 * pad - filter_size + 1;

    const float* data_im = input.data();
    float* data_col = output.data();

    for (int channel = channels; channel--; data_im += BOARD_SQUARES) {
        for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
            for (unsigned int kernel_col = 0; kernel_col < filter_size; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (unsigned(input_row) < height) {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (unsigned(input_col) < width) {
                                *(data_col++) =
                                    data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col++;
                        }
                    } else {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    }
                    input_row++;
                }
            }
        }
    }
}

template <>
void im2col<1>(const int channels,
               const std::vector<float>& input,
               std::vector<float>& output) {
    auto outSize = size_t{channels * static_cast<size_t>(BOARD_SQUARES)};
    assert(output.size() == outSize);
    std::copy(begin(input), begin(input) + outSize, begin(output));
}

#endif
