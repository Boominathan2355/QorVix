# Wrap glslc's `-mfmt=c` output ("{ 0x07230203, ... }") into a named C array so it matches the
# header glslang's --vn produces: `const uint32_t <VN>[] = { ... };`. Used only on the glslc
# fallback path; glslang emits the named array directly.
file(READ "${IN}" _body)
file(WRITE "${OUT}" "const uint32_t ${VN}[] = ${_body};\n")
