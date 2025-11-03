set(COMPILED_SPV_FILES "")
foreach(GLSL_FILE ${VK_GLSL_SHADERS})
   get_filename_component(FILE_NAME ${GLSL_FILE} NAME)
   set(SPV_FILE "${VK_SHADER_BINARY_DIR}/${FILE_NAME}.spv")
   add_custom_command(
      OUTPUT ${SPV_FILE}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${VK_SHADER_BINARY_DIR}
      COMMAND ${GLSLANG_VALIDATOR} -V ${GLSL_FILE} -o ${SPV_FILE}
      DEPENDS ${GLSL_FILE}
      COMMENT "Compiling ${FILE_NAME} to SPIR-V for Vulkan"
      VERBATIM
   )
   list(APPEND COMPILED_SPV_FILES ${SPV_FILE})
endforeach()
