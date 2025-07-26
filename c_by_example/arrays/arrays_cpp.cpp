#include <array>
#include <iostream>

int main() {
  std::array<int, 5> array;

  for (size_t i = 0; i < array.size(); i++) {
    array[i] = i;
  }

  std::cout << "array[2] = " << array[2] << std::endl;
  return 0;
}
