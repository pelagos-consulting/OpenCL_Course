__kernel void xcorr(
        __global float *src, 
        __global float *dst,
        __global float *kern,
        unsigned int len0_src,
        unsigned int len1_src, 
        unsigned int pad0_l,
        unsigned int pad0_r,
        unsigned int pad1_l,
        unsigned int pad1_r      
    ) {

    // get the coordinates
    size_t i0 = get_global_id(1);
    size_t i1 = get_global_id(0);

    //// Complete the correlation kernel //// 

}
