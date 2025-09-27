[TODO] Use the man pages to read the documentation for the “strace” command
  [SUB-TODO] Find out the option that allows you to count the number of occurrences of each system call (Hint: Might be Statistics)

[TODO]  Create a c program called “mycat”
      SPECS :   1. main routine should accept an input parameter from the user an input file name
                  (passed to your program when it was invoked from the shell)
                2. use unix calls for opening, closing and reading files i.e. use open instead of fopen, etc. you can use either system call or c call for printing
                3. program will sleep/wait for a random number of range[1,5] seconds between printing the process ID
                   and printing the content of the input file.
                4. example of how the program should be invoked from the shell where input.txt is a file that already exists 
                   (you may create an input file with a few lines to test your code) : 

```bash
mycat input.txt
```
                5. if a path is not provided in filenames, then it’s assumed that a file is located at the same directory as
                   the working directory of your program
                6. implement error handling, e.g., input param, file operations & learn to write prod level c tests

      INPUT :  input.txt file

      OUTPUT:   1. First line   :  print the process ID of your (running) program, an integer (Hint: use function getpid())
                2. Second line* :  2.1. print the contents of the input file  
                                   2.2. print the process ID and content of the input file.

      TEST  :   1. if file cannot be found
                2. if file is empty
                3. if file doesn't have permissions?
                4. too many arguments
                5. check character encoding?


[TODO] After developing your program, invoke using strace and then answer the following :
    [SUB-TODO] 1. system call names for getting:
                1.1. process ID
                1.2. opening a file
                1.3. closing a file
                1.4. reading a file
                1.5. printing to the console
                1.6. sleeping
    [SUB-TODO] 2. number of system calls (i.e. how many times each was called) for:  
                2.1. opening a file
                2.2. closing a file
                2.3. reading a file
    [SUB-TODO] 3. number of system calls for printing to the screen ( use strace options or grep)
    [SUB-TODO] 4. value of the file descriptor of the read file (review the lecture slides to find out what this means)


[NOTES] 1. include your answers to questions and the strace log in your submitted .pdf file.
      2. create a text file and use it to test your program - like put some shakespeare novel starting paragraph
      3. use the man pages to learn how to use POSIX API library functions (and the necessary include files) and/or
         unix commands and its various optional arguments (e.g. strace, especially for counting), e.g.:

          man strace   // gets info from section 1, user’s manual
          man getpid.2 // section 2 is programmer’s manual


