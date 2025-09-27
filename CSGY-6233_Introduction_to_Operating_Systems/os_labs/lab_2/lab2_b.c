#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
  int source_file;
  char buffer[BUFFER_SIZE];
  ssize_t bytes_read;
  int sleep_time;
  pid_t process_id;

  source_file = open(argv[1], O_RDONLY);
  if (source_file == -1) {
    printf("error in opening file \n");
    return 1;
  }
  process_id = getpid();
  printf("process-id : %d\n", process_id);

  srand(time(NULL));
  sleep_time = (rand() % 5) + 1;
  sleep(sleep_time);

  while ((bytes_read = read(source_file, buffer, BUFFER_SIZE)) > 0) {
    if (write(STDOUT_FILENO, buffer, bytes_read) == -1) {
      printf("error in writing to stdout \n");
      close(source_file);
      return 1;
    }
  }

  if (bytes_read == -1) {
    printf("error in reading file \n");
    close(source_file);
    return 1;
  }

  if (close(source_file) == -1) {
    printf("error in closing file \n");
    return 1;
  }
  return 0;
}
