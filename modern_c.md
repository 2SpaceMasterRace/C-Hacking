# Learnings

## Tips

1. You can explain a complicated declaration in English using cdelc.
```bash
  # Ubuntu
  apt install cdecl 

  # Arch Linux
  yay -S cdecl

  echo "explain char *(*pfpc)()" | cdecl
```


## Data Structures / Types

1. Type Qualifiers

const will qualify a given type as a constant value (it cannot change, or in other words, it is not a variable).

You can make a pointer constant, a non-pointer data type constant, and a constant pointer to a constant data type (a "double const"). You probably want to default to the double const, as this tells the compiler that both the pointer, and the value it is pointing at, should not be modified.


```c
           #include <stdint.h>
           int8_t bob = 42;                   // a variable
           int8_t const bob_c = 42;           // a constant data type
           int8_t * dobbs_1 = &bob;           // a regular pointer; both pointer and data can change
           int8_t const * dobbs_2 = &bob;     // a constant pointer to a variable; the 
           int8_t const * const dobbs = &bob; // a constant pointer to a constant data type
``` 


You can read declarations backwards/spirally to make sense of this.
 

2. Do not use int, char, short, long, unsigned. Use Fixed-width types whenever possible.
  2.1. Use #include <stdint.h> in your code
  2.2. Use a specific integer width type like int8_t foo;. This is more portable than hoping your compiler and CPU architecture give you what you were expecting for int.
  2.3. If you only need a minimum integer width, use the fast variants, like int_fast8_t. These may be larger than expected but will be faster for your platform.
  2.4. Don't use char if you're just trying to do random unsigned byte manipulations; use uint8_t.
  2.5. Booleans: For C99 and later, include #include <stdbool.h> and bool my_var = true; or bool my_var = false; (the C standard is basically "zero for false and non-zero for true")


## Memory Management
1. Learn to group allocations (reduces the amount of times you have to call free() and can improve performance) - use arena allocation 
2. Avoid dynamic memory allocation by using the stack
3. If you use malloc, always #include <stdlib.h>. If you don't, you will get casting errors, and the compiler will assume malloc returns an int.


## Managing polymorphic data
1. Use tagged unions
2. Organize data and logic better by prefixing functions that are associated with data by the name of the data type: cat_meow(&cat)
3. Generics: non-intrusive containers, function pointers or _Generics
    3.1. Non-intrusive data structures: you can get around not having generics by passing in type-specific information as parameters
    3.2. Function pointers: A way to inject logic, make sure its inlinable if performance matteres
    3.3. _Generic: When paired with a wrapper macro, can let you select which function to call at compile-time

