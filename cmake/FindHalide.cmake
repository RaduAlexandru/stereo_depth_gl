# - Try to find the HALIDE library
# Once done this will define
#
#  HALIDE_FOUND - system has HALIDE
#  HALIDE_INCLUDE_DIR - **the** HALIDE include directory
#  HALIDE_INCLUDE_DIRS - HALIDE include directories
#  HALIDE_SOURCES - the HALIDE source files

if(NOT HALIDE_FOUND)
	find_path(HALIDE_INCLUDE_DIR
		NAMES Halide.h
	   	PATHS /media/alex/Data/Programs_linux/Halide/build/include
		DOC "The Halide include directory"
		NO_DEFAULT_PATH)

	if(HALIDE_INCLUDE_DIR)
	   set(HALIDE_FOUND TRUE)
	   set(HALIDE_INCLUDE_DIRS ${HALIDE_INCLUDE_DIR})
	else()
	   message("+-------------------------------------------------+")
	   message("| Halide include not found						  |")
	   message("+-------------------------------------------------+")
	   message(FATAL_ERROR "")
	endif()


	#library
	find_library(HALIDE_LIBRARY_DIR
		NAMES libHalide.so
	   	HINTS /media/alex/Data/Programs_linux/Halide/build/lib/
		DOC "The Halide lib directory"
		NO_DEFAULT_PATH)

	if(HALIDE_LIBRARY_DIR)
	   set(HALIDE_LIBRARIES "${HALIDE_LIBRARY_DIR}")
	else()
	  message("+-------------------------------------------------+")
	  message("| Halide library not found                   	 |")
	  message("+-------------------------------------------------+")
	  message(FATAL_ERROR "")
	endif()

endif()
