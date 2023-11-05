# MCUCompiler.jl

MCUCompiler.jl is a julia-to-MCU compiler support library. Its used for compiling julia source code targeting microcontrollers.

To install this package, use `]add https://github.com/Seelengrab/MCUCompiler.jl`

## Requirements for use

For some backends (like ARM), you will have to build Julia yourself, since the LLVM version that is shipped with Julia does not come
with that backend enabled.

Change this in `deps/llvm.mk` of your local clone of the [julialang repository](https://github.com/JuliaLang/julia):

```patch
diff --git a/deps/llvm.mk b/deps/llvm.mk
index 5d297b6c36..3a7720bd71 100644
--- a/deps/llvm.mk
+++ b/deps/llvm.mk
@@ -64,7 +64,7 @@ endif
 LLVM_LIB_FILE := libLLVMCodeGen.a
 
 # Figure out which targets to build
-LLVM_TARGETS := host;NVPTX;AMDGPU;WebAssembly;BPF;AVR
+LLVM_TARGETS := host;NVPTX;AMDGPU;WebAssembly;BPF;AVR;ARM
 LLVM_EXPERIMENTAL_TARGETS :=
 
 LLVM_CFLAGS :=
```

And add this to `Make.user`:

```text
USE_BINARYBUILDER_LLVM=0
```

The first patch enables the `ARM` backend to be built, the second tells the julia build process to build a local version of LLVM.

Finally, build julia by running `make`. Since you're also going to build LLVM, this can take some time - a full build of LLVM takes about
45 minutes on my laptop, running a i7-6600U.

## Usage

The usage workflow generally is like this:

 * Write code to be compiled
 * Define a target to compile for - take a look at how its done for AVR in `src/arduino.jl`.
 * Build & flash according to your platform.

See the API section below for more utility functionality. `@code_llvm dump_module=true myproject.main()` is generally very useful for
debugging purposes, as are JET.jl and Cthulhu.jl. Don't try to run functions intended to be compiled to AVR in a regular julia session - 
your program will probably segfault.

## API

The main entry point for your compilation process is the `build` function. Read its docstring for more information.
