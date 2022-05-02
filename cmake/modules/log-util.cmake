function(typeart_target_define_file_basename targetname)
  get_target_property(source_files "${targetname}" SOURCES)

  foreach(sourcefile ${source_files})
    get_property(compile_defs
      SOURCE "${sourcefile}"
      PROPERTY COMPILE_DEFINITIONS
    )

    get_filename_component(basename "${sourcefile}" NAME)

    list(APPEND compile_defs
      "LOG_BASENAME_FILE=\"${basename}\""
    )

    set_source_files_properties("${sourcefile}"
      PROPERTIES COMPILE_DEFINITIONS ${compile_defs}
    )
  endforeach()
endfunction()
