module Arduino

using avr_binutils_jll
using avrdude_jll

import ..MCUCompiler: MCUCompiler, mcu_job, triple, build_vectors, link, postprocess
import GPUCompiler: AbstractCompilerParams

#####
# Compiler Target
#####

const ArduinoTarget = MCUCompiler.MCUTarget{:Arduino}

struct ArduinoParams <: AbstractCompilerParams
    name::String
end

triple(::ArduinoTarget) = "avr-unknown-unkown"

avr_job(@nospecialize(func), @nospecialize(types), platform=ArduinoTarget(ArduinoParams("$(nameof(func))"))) = mcu_job(func, types, platform)

function build_vectors(::ArduinoTarget, asm_path, obj_path)
    open(vectorasm_path, "w") do io
        println(io, """
          .vectors:
                rjmp main
          """)
        # TODO: `println` additional calls for interrupt vectors
    end
    avr_as() do bin
        run(`$bin -o $vectorobj_path $vectorasm_path`)
    end
end

function link(::ArduinoTarget, elf, vectors, obj)
    avr_ld() do bin
        run(`$bin -v -o $elf $vectors $obj`)
    end
end

function postprocess(at::ArduinoTarget, buildpath)
    mainhex_name = string(at.params.name, ".hex")
    builthex_name = joinpath(buildpath, mainhex_name)
    avr_objcopy() do bin
        run(`$bin -O ihex $builtelf_name $builthex_name`)
    end
end

"""
    list_mcus()

List the microcontrollers supported by `avrdude`.
"""
function list_mcus()
    avrdude() do bin
        run(Cmd(`$bin -p \?`; ignorestatus=true))
    end
    nothing
end

"""
    avr_flash(path, bin, partno
          ; clear=true, verify=true, programmer=:arduino)

Flash the binary `bin` to the device connected at `path`.
`partno` specifies the microcontroller that will be flashed.

 * `clear` specifies whether to clear the flash ROM of the device
 * `verify` tells the programmer to verify the written data
 * `programmer` specifies the programmer to use for flashing

!!! warn "Defaults"
    This is intended as a convenient interface to `avrdude` from `avrdude_jll`.
    For more complex configurations, consider using the JLL directly.
    The defaults specified here are only tested for an Arduino Ethernet with an ATmega328p.

!!! warn "Warranty"
    Using this to flash your device is not guaranteed to succeed and no warranty of any kind
    is given. Use at your own risk.
"""
function flash(path, binpath, partno; clear=true, verify=true, programmer=:arduino)
    ispath(path) || throw(ArgumentError("`$path` is not a path."))
    isfile(binpath) || throw(ArgumentError("`$binpath` is not a file."))
    flasharg = ':' in binpath ? `flash:w:$binpath:a` : `$binpath`
    verifyarg = verify ? `` : `-V`
    cleararg = clear ? `` : `-D`
    avrdude() do bin
        run(`$bin $verifyarg -c $programmer -p $partno -P $path $cleararg -U $flasharg`)
    end
    nothing
end

end
