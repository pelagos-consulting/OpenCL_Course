#include <cstdio>

// Helper application
#include "cl_helper.hpp"

// Main program
int main(int argc, char** argv) {
    
    // This is a comment
    char a='s'; // Using char as a character
   
    { // Starting a code block
        // Declaring integers
        char a_i=1;         // Using char as an integer
        short d_i=4;        // 16 bit
        int b_i=2;          // 32 bit          
        unsigned int c_i=3; // 32 bit
        long e_i = 5;       // 64 bit
    } 

    // Arithmetic with chars
    char b = a+1;
    printf("%i\n", b); // print b with the memory interpreted as an integer
    printf("%c\n", b); // print b with the memory interpreted as a character

    // Declaring floating point value
    float x=5.0;
    double y=6.0;
    long double z=7.0;

    // Printing floats
    printf("%f %f\n", x, y); // Print x and y to the screen with their memory interpreted as floats
    printf("%i %i\n", y, x); // Print y and x to the screen with their memory interpreted as integers



}
