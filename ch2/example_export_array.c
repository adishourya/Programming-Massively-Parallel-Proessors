#include <stdio.h>
#include <stdlib.h>

int main() {
  // Define dimensions and allocate memory for the array
  int rows = 3;
  int cols = 3;
  float data[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}};

  // Open a binary file for writing
  FILE *file = fopen("array_data.bin", "wb");
  if (file == NULL) {
    perror("Unable to open file");
    return EXIT_FAILURE;
  }

  // Write the array data to the binary file
  size_t result = fwrite(data, sizeof(float), rows * cols, file);
  if (result != rows * cols) {
    perror("Error writing file");
    fclose(file);
    return EXIT_FAILURE;
  }

  // Close the file
  fclose(file);

  return EXIT_SUCCESS;
}
