# AVRCompiler.jl

AVRCompiler.jl is a julia-to-AVR compiler support library. Its used for compiling julia source code to AVR assembly.

## Requirements for use

Since julia currently does not come with the AVR backend enabled, you will have to build julia yourself.

Change this in `deps/llvm.mk` of your local clone of the [julialang repository](https://github.com/JuliaLang/julia):

```patch
diff --git a/deps/llvm.mk b/deps/llvm.mk
index 5d297b6c36..3a7720bd71 100644
--- a/deps/llvm.mk
+++ b/deps/llvm.mk
@@ -64,7 +64,7 @@ endif
 LLVM_LIB_FILE := libLLVMCodeGen.a
 
 # Figure out which targets to build
-LLVM_TARGETS := host;NVPTX;AMDGPU;WebAssembly;BPF
+LLVM_TARGETS := host;NVPTX;AMDGPU;WebAssembly;BPF;AVR
 LLVM_EXPERIMENTAL_TARGETS :=
 
 LLVM_CFLAGS :=
```

And add this to `Make.user`:

```text
USE_BINARYBUILDER_LLVM=0  
```

The first patch enables the `AVR` backend to be built, the second tells the julia build process to build a local version of LLVM.

Finally, build julia by running `make`.

## Usage

The usage workflow generally is like this:

 * Write code to be compiled
 * Compile it like 
   * `obj = build_obj(myproject.main, ())`
 * Write it to disk
   * `write("out/jl_out.o", obj)`
 * Link it from your shell
   * `avr-ld -o jl_out.elf jl_out.o`
 * Convert the `.elf` to the `ihex` format the µc expects
   * `avr-objcopy -O ihex jl_out.elf jl_out.hex`
 * Flash the program onto your microcontroller
   * `avrdude -V -c arduino -p <MICROCONTROLLER NAME> -P <MOUNT PATH OF YOUR MICROCONTROLLER> -U flash:w:jl_out.hex`

These commands require an AVR buildchain to be installed on your device. This is most commonly called `avr-binutils`, but may vary
depending on your platform.

The commands written above may not work with your exact versions of these utilities - adapt accordingly.

See the API section below for more utility functionality. `@code_llvm dump_module=true myproject.main()` is generally very useful for
debugging purposes, as are JET.jl and Cthulhu.jl. Don't try to run functions intended to be compiled to AVR in a regular julia session - 
your program will probably segfault.

## API

AVRCompiler.jl currently provides 3 functions for use/inspection:

 * `native_job(f, types, params)`
   * `f`: The function to compile - most commonly `main`, taking no arguments
   * `types`: The argument types to the function - most commonly the empty tuple `()`
   * `params`: An `ArduinoParams` object containing metadata for the build, like the name to be used for identification in the binary
   * Returns a `GPUCompiler.Compilerjob`, which is the job context that will be used for compilation
 * `build_ir(job, f, types; optimize=true)`
   * `job`: The `CompilerJob` object to be used for compilation context
   * `f`: The function to compile - most commonly `main`, taking no arguments
   * `types`: The argument types to the function - most commonly the empty tuple `()`
   * `optimize`: A keyword argument specifying whether the IR should be optimized by GPUCompiler - currently mandatory to be `true`, if you want the build to remove unused references to the julia runtime (which we can't use on a microcontroller)
   * Returns a tuple of LLVM-IR and IR-metadata
 * `build_obj(func, types[, params]; strip=true, validate=true)`
   * `func`: The function to compile
   * `types`: The types of the arguments to compile for func
   * `params`: An `ArduinoParams` object containing metadata for the build - defaults to `ArduinoParams("unnamed")`
   * `strip`: A keyword argument specifying whether the binary should be stripped of symbols - defaults to `true`
   * `validate`: A keyword argument specifying whether LLVM should check the produce IR for being correct - defaults to `true`
   * Returns the built artifact as an object file in form of a string. Can be written to disk like `write(outpath, obj)`
* `builddump(func, args)`
   * `func`: The function to compile
   * `args`: The argument types to compile `func` with
   * Compiles & ultimately prints a decompiled dump of the unlinked object file
   * Requires `avr-objdump` to be installed on your system (most commonly installable under the name `avr-binutils`)