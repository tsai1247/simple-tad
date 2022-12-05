#include <iostream>

struct HiC {
    std::string chr;
    int bin1;
    int bin2;
    double rescaled_intensity;
    int diag_offset;
    int dist;
};