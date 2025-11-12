#include <cstdint>
#include <iostream>
#include <string>
#include <immintrin.h>
#include "avx.hpp"
#include "avx.cpp"

using namespace std;


// Print by-reference to the whole array (const: we only read)
void print_result(const string& name, avx::Result result, int8_t (&tmp)[26]) {
    cerr << name << " -> any_negative=" << std::boolalpha << result.any_negative << " all_zero=" << std::boolalpha << result.all_zero << " : ";
    for (int i = 0; i < 26; ++i) cerr << int(tmp[i]) << ' ';  // cast so it prints numbers, not int8_ts
    cerr << endl;
}

// Take array refs so sizes don’t decay; write result into tmp
void test_result(const string& name,
                 int8_t (&tmp)[26],
                 int8_t (&input)[26],
                 int8_t (&other)[26]) {
    auto result = avx::compare (
		input, other, tmp
	);
    print_result(name, result, tmp);
}

int main() {
    int8_t input[26] = {3,2,1,0};
    int8_t dictionary_word_perfect_fit[26] = {3,2,1,0};
    int8_t dictionary_word_does_not_fit[26] = {6,2,1,0};
    int8_t dictionary_word_fits_with_leftover[26] = {2,1,0};
    int8_t tmp[26];

    test_result("Perfect fit", tmp, input, dictionary_word_perfect_fit);
    test_result("Does not fit", tmp, input, dictionary_word_does_not_fit);
    test_result("Fits w/ leftover", tmp, input, dictionary_word_fits_with_leftover);
}
