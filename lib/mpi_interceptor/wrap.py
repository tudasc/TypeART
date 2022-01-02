#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
#################################################################################################
# Copyright (c) 2010, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Todd Gamblin, tgamblin@llnl.gov.
# LLNL-CODE-417602
# All rights reserved.
#
# This file is part of Libra. For details, see http://github.com/tgamblin/libra.
# Please also read the LICENSE file for further information.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the disclaimer below.
#  * Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the disclaimer (as noted below) in the documentation and/or other materials
#    provided with the distribution.
#  * Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse
#    or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################################
from __future__ import print_function
usage_string = \
'''Usage: wrap.py [-fgd] [-i pmpi_init] [-c mpicc_name] [-o file] wrapper.w [...]
 Python script for creating PMPI wrappers. Roughly follows the syntax of
   the Argonne PMPI wrapper generator, with some enhancements.
 Options:"
   -d             Just dump function declarations parsed out of mpi.h
   -f             Generate fortran wrappers in addition to C wrappers.
   -g             Generate reentry guards around wrapper functions.
   -s             Skip writing #includes, #defines, and other front-matter (for non-C output).
   -c exe         Provide name of MPI compiler (for parsing mpi.h).  Default is \'mpicc\'.
   -I dir         Provide an extra include directory to use when parsing mpi.h.
   -i pmpi_init   Specify proper binding for the fortran pmpi_init function.
                  Default is \'pmpi_init_\'.  Wrappers compiled for PIC will guess the
                  right binding automatically (use -fPIC when you compile dynamic libs).
   -o file        Send output to a file instead of stdout.
   -w             Do not print compiler warnings for deprecated MPI functions.
                  This option will add macros around {{callfn}} to disable (and
                  restore) the compilers diagnostic functions, if the compiler
                  supports this functionality.

 by Todd Gamblin, tgamblin@llnl.gov
'''
import tempfile, getopt, subprocess, sys, os, re, types, itertools

# Default values for command-line parameters
mpicc = 'mpicc'                    # Default name for the MPI compiler
includes = []                      # Default set of directories to inlucde when parsing mpi.h
pmpi_init_binding = "pmpi_init_"   # Default binding for pmpi_init
pmpi_init_thread_binding = "pmpi_init_thread_"   # Default binding for pmpi_init_thread
output_fortran_wrappers = False    # Don't print fortran wrappers by default
output_guards = False              # Don't print reentry guards by default
skip_headers = False               # Skip header information and defines (for non-C output)
dump_prototypes = False            # Just exit and dump MPI protos if false.
static_dir = None                  # Directory to write source files for static lib to
ignore_deprecated = False          # Do not print compiler warnings for deprecated MPI functions

# Possible legal bindings for the fortran version of PMPI_Init()
pmpi_init_bindings = ["PMPI_INIT", "pmpi_init", "pmpi_init_", "pmpi_init__"]
# Possible legal bindings for the fortran version of PMPI_Init_thread()
pmpi_init_thread_bindings = ["PMPI_INIT_THREAD", "pmpi_init_thread", "pmpi_init_thread_", "pmpi_init_thread__"]

# Possible function return types to consider, used for declaration parser.
# In general, all MPI calls we care about return int.  We include double
# to grab MPI_Wtick and MPI_Wtime, but we'll ignore the f2c and c2f calls
# that return MPI_Datatypes and other such things.
# MPI_Aint_add and MPI_Aint_diff return MPI_Aint, so include that too.
rtypes = ['int', 'double', 'MPI_Aint' ]

# If we find these strings in a declaration, exclude it from consideration.
exclude_strings = [ "c2f", "f2c", "typedef", "MPI_T_", "MPI_Comm_spawn" ]

# Regular expressions for start and end of declarations in mpi.h. These are
# used to get the declaration strings out for parsing with formal_re below.
begin_decl_re = re.compile("(" + "|".join(rtypes) + ")\s+(MPI_\w+)\s*\(")
exclude_re =    re.compile("|".join(exclude_strings))
end_decl_re =   re.compile("\).*\;")

# Regular Expression for splitting up args. Matching against this
# returns three groups: type info, arg name, and array info
formal_re = re.compile(
    "\s*(" +                       # Start type
    "(?:const)?\s*" +              # Initial const
    "\w+"                          # Type name (note: doesn't handle 'long long', etc. right now)
    ")\s*(" +                      # End type, begin pointers
    "(?:\s*\*(?:\s*const)?)*" +    # Look for 0 or more pointers with optional 'const'
    ")\s*"                         # End pointers
    "(?:(\w+)\s*)?" +              # Argument name. Optional.
     "(\[.*\])?\s*$"               # Array type.  Also optional. Works for multidimensions b/c it's greedy.
    )

# Fortran wrapper suffix
f_wrap_suffix = "_fortran_wrapper"

# Initial includes and defines for wrapper files.
wrapper_includes = '''
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */



'''

fortran_wrapper_includes = '''
#ifdef MPICH_HAS_C2F
_EXTERN_C_ void *MPIR_ToPointer(int);
#endif // MPICH_HAS_C2F

#if defined(MPI_STATUS_SIZE) && MPI_STATUS_SIZE > 0
#define MPI_F_STATUS_SIZE MPI_STATUS_SIZE
#else
void get_mpi_f_status_size___(int*);
static inline int __get_f_status_size(){int size; get_mpi_f_status_size___(&size); return size;}
#define MPI_F_STATUS_SIZE __get_f_status_size()
#endif

void char_p_f2c(const char* fstr, int len, char** cstr)
{
  const char* end;
  int i;

  /* Leading and trailing blanks are discarded. */

  end = fstr + len - 1;

  for (i = 0; (i < len) && (' ' == *fstr); ++i, ++fstr) {
    continue;
  }

  if (i >= len) {
    len = 0;
  } else {
    for (; (end > fstr) && (' ' == *end); --end) {
      continue;
    }

    len = end - fstr + 1;
  }

  /* Allocate space for the C string, if necessary. */

  if (*cstr == NULL) {
    if ((*cstr = malloc(len + 1)) == NULL) {
      return;
    }
  }

  /* Copy F77 string into C string and NULL terminate it. */

  if (len > 0) {
    strncpy(*cstr, fstr, len);
  }
  (*cstr)[len] = '\\0';
}

void char_p_c2f(const char* cstr, char* fstr, int len)
{
  int i;

  strncpy(fstr, cstr, len);
  for (i = strnlen(cstr, len); i < len; ++i) {
    fstr[i] = ' ';
  }
}

#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PGI) || defined(_CRAYC)
#if defined(__GNUC__)
#define WEAK_POSTFIX __attribute__ ((weak))
#else
#define WEAK_POSTFIX
#define USE_WEAK_PRAGMA
#endif
/* OpenMPI */
_EXTERN_C_ MPI_Fint mpi_fortran_in_place WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPI_FORTRAN_IN_PLACE WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpi_fortran_in_place_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPI_FORTRAN_IN_PLACE_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpi_fortran_in_place__ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPI_FORTRAN_IN_PLACE__ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpi_fortran_bottom WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPI_FORTRAN_BOTTOM WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpi_fortran_bottom_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPI_FORTRAN_BOTTOM_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpi_fortran_bottom__ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPI_FORTRAN_BOTTOM__ WEAK_POSTFIX;
/* MPICH 2 */
_EXTERN_C_ MPI_Fint MPIFCMB3 WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpifcmb3 WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPIFCMB3_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpifcmb3_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPIFCMB3__ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpifcmb3__ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPIFCMB4 WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpifcmb4 WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPIFCMB4_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpifcmb4_ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint MPIFCMB4__ WEAK_POSTFIX;
_EXTERN_C_ MPI_Fint mpifcmb4__ WEAK_POSTFIX;
/* Argonne Fortran MPI wrappers */
_EXTERN_C_ void *MPIR_F_MPI_BOTTOM WEAK_POSTFIX;
_EXTERN_C_ void *MPIR_F_MPI_IN_PLACE WEAK_POSTFIX;
_EXTERN_C_ void *MPI_F_MPI_BOTTOM WEAK_POSTFIX;
_EXTERN_C_ void *MPI_F_MPI_IN_PLACE WEAK_POSTFIX;

/* MPICH 2 requires no special handling - MPI_BOTTOM may (must!) be passed through as-is. */
#define IsBottom(x) ((x) == (void *) &mpi_fortran_bottom ||   \\
                     (x) == (void *) &MPI_FORTRAN_BOTTOM ||   \\
                     (x) == (void *) &mpi_fortran_bottom_ ||  \\
                     (x) == (void *) &MPI_FORTRAN_BOTTOM_ ||  \\
                     (x) == (void *) &mpi_fortran_bottom__ || \\
                     (x) == (void *) &MPI_FORTRAN_BOTTOM__)
#define IsInPlace(x) ((x) == (void *) &mpi_fortran_in_place ||   \\
                      (x) == (void *) &MPI_FORTRAN_IN_PLACE ||   \\
                      (x) == (void *) &mpi_fortran_in_place_ ||  \\
                      (x) == (void *) &MPI_FORTRAN_IN_PLACE_ ||  \\
                      (x) == (void *) &mpi_fortran_in_place__ || \\
                      (x) == (void *) &MPI_FORTRAN_IN_PLACE__ || \\
                      (x) == (void *) &MPIFCMB4 ||               \\
                      (x) == (void *) &mpifcmb4 ||               \\
                      (x) == (void *) &MPIFCMB4_ ||              \\
                      (x) == (void *) &mpifcmb4_ ||              \\
                      (x) == (void *) &MPIFCMB4__ ||             \\
                      (x) == (void *) &mpifcmb4__ ||             \\
                      (x) == MPIR_F_MPI_IN_PLACE ||              \\
                      (x) == MPI_F_MPI_IN_PLACE)

#ifdef USE_WEAK_PRAGMA
#pragma weak mpi_fortran_in_place
#pragma weak MPI_FORTRAN_IN_PLACE
#pragma weak mpi_fortran_in_place_
#pragma weak MPI_FORTRAN_IN_PLACE_
#pragma weak mpi_fortran_in_place__
#pragma weak MPI_FORTRAN_IN_PLACE__
#pragma weak mpi_fortran_bottom
#pragma weak MPI_FORTRAN_BOTTOM
#pragma weak mpi_fortran_bottom_
#pragma weak MPI_FORTRAN_BOTTOM_
#pragma weak mpi_fortran_bottom__
#pragma weak MPI_FORTRAN_BOTTOM__
/* MPICH 2 */
#pragma weak MPIFCMB3
#pragma weak mpifcmb3
#pragma weak MPIFCMB3_
#pragma weak mpifcmb3_
#pragma weak MPIFCMB3__
#pragma weak mpifcmb3__
#pragma weak MPIFCMB4
#pragma weak mpifcmb4
#pragma weak MPIFCMB4_
#pragma weak mpifcmb4_
#pragma weak MPIFCMB4__
#pragma weak mpifcmb4__
/* Argonne Fortran MPI wrappers */
#pragma weak MPIR_F_MPI_BOTTOM
#pragma weak MPIR_F_MPI_IN_PLACE
#pragma weak MPI_F_MPI_BOTTOM
#pragma weak MPI_F_MPI_IN_PLACE
#endif

#if defined(MPICH_NAME) && (MPICH_NAME == 1) /* MPICH has no MPI_IN_PLACE */
#define BufferC2F(x) (IsBottom(x) ? MPI_BOTTOM : (x))
#else
#define BufferC2F(x) (IsBottom(x) ? MPI_BOTTOM : (IsInPlace(x) ? MPI_IN_PLACE : (x)))
#endif /* defined(MPICH_NAME) && (MPICH_NAME == 1) */

#else
#define BufferC2F(x) (x)
#endif /* defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__PGI) || defined(_CRAYC) */

'''

wrapper_main_pmpi_init_decls ='''
#if (defined(PIC) || defined(__PIC__))
/* For shared libraries, declare these weak and figure out which one was linked
   based on which init wrapper was called.  See mpi_init wrappers.  */
#pragma weak pmpi_init
#pragma weak PMPI_INIT
#pragma weak pmpi_init_
#pragma weak pmpi_init__
#pragma weak pmpi_init_thread
#pragma weak PMPI_INIT_THREAD
#pragma weak pmpi_init_thread_
#pragma weak pmpi_init_thread__
#endif /* PIC */

_EXTERN_C_ void pmpi_init(MPI_Fint *ierr);
_EXTERN_C_ void PMPI_INIT(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init__(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_thread(MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierr);
_EXTERN_C_ void PMPI_INIT_THREAD(MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_thread_(MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_thread__(MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierr);

'''

# Declarations from the .w file to be repeated in every source file.
declarations = None

# Macros used to suppress MPI deprecation warnings.
wrapper_diagnosics_macros = '''
/* Macros to enable and disable compiler warnings for deprecated functions.
 * These macros will be used to silent warnings about deprecated MPI functions,
 * as these should be wrapped, even if they are deprecated.
 *
 * Note: The macros support GCC and clang compilers only. For other compilers
 *       just add similar macros for your compiler.
 */
#ifndef WRAP_MPI_CALL_PREFIX
#if defined(__clang__)
#define WRAP_MPI_CALL_PREFIX        \\
  _Pragma("clang diagnostic push"); \\
  _Pragma("clang diagnostic ignored \\"-Wdeprecated-declarations\\"");
#define WRAP_MPI_CALL_POSTFIX _Pragma("clang diagnostic pop");
#elif defined(__INTEL_COMPILER)
#define WRAP_MPI_CALL_PREFIX \\
  _Pragma("warning push");   \\
  _Pragma("warning(disable:1786)");
#define WRAP_MPI_CALL_POSTFIX _Pragma("warning pop")
#elif (defined(__GNUC__) && \\
  ((__GNUC__ == 4 && __GNUC_MINOR__ >= 6) || __GNUC__ > 4))
#define  WRAP_MPI_CALL_PREFIX     \\
  _Pragma("GCC diagnostic push"); \\
  _Pragma("GCC diagnostic ignored \\"-Wdeprecated-declarations\\"");
#define WRAP_MPI_CALL_POSTFIX _Pragma("GCC diagnostic pop");
#else
#define WRAP_MPI_CALL_PREFIX
#define WRAP_MPI_CALL_POSTFIX
#endif
#endif

'''

# Default modifiers for generated bindings
default_modifiers = ["_EXTERN_C_"]  # _EXTERN_C_ is #defined (or not) in wrapper_includes. See above.

# Set of MPI Handle types
mpi_handle_types = set(["MPI_Comm", "MPI_Errhandler", "MPI_File", "MPI_Group", "MPI_Info",
                        "MPI_Op", "MPI_Request", "MPI_Status", "MPI_Datatype", "MPI_Win" ])

# MPI Calls that have array parameters, and mappings from the array parameter
# positions to the position of the 'count' parameters that determine their size.
mpi_array_calls = {
    "MPI_Startall"           : { 1:0 },
    "MPI_Testall"            : { 1:0, 3:0 },
    "MPI_Testany"            : { 1:0 },
    "MPI_Testsome"           : { 1:0, 4:0 },
    "MPI_Type_create_struct" : { 3:0 },
    "MPI_Type_get_contents"  : { 6:1 },
    "MPI_Type_struct"        : { 3:0 },
    "MPI_Waitall"            : { 1:0, 2:0 },
    "MPI_Waitany"            : { 1:0 },
    "MPI_Waitsome"           : { 1:0, 4:0 }
}

# MPI Calls that have an array index output parameter (i.e. "int *index") and
# position of that 'index' parameter.  Needs special casing when wrapping Fortran
# calls (arrays '1' indexed) to C (arrays '0' indexed). May also be MPI_UNDEFINED.
mpi_array_index_output_calls = {
    "MPI_Testany"            : 2,
    "MPI_Waitany"            : 2
}

# MPI Calls that have an array of array-indices as an output parameter (i.e.
# "int *array_of_indices") and mappings from the position of that 'array_of_indices'
# to the position of the output parameter that determines its size.
# Needs special casing when wrapping Fortran calls (arrays '1' indexed) to C
# (arrays '0' indexed).
mpi_array_index_array_output_calls = {
    "MPI_Testsome"           : { 3:2 },
    "MPI_Waitsome"           : { 3:2 }
}

def find_matching_paren(string, index, lparen='(', rparen=')'):
    """Find the closing paren corresponding to the open paren at <index>
       in <string>.  Optionally, can provide other characters to match on.
       If found, returns the index of the matching parenthesis.  If not found,
       returns -1.
    """
    if not string[index] == lparen:
        raise ValueError("Character at index %d is '%s'. Expected '%s'"
                         % (index, string[index], lparen))
    index += 1
    count = 1
    while index < len(string) and count > 0:
        while index < len(string) and string[index] not in (lparen, rparen):
            index += 1
        if string[index] == lparen:
            count += 1
        elif string[index] == rparen:
            count -= 1

    if count == 0:
        return index
    else:
        return -1


def find_matching_paren(string, index, lparen='(', rparen=')'):
    """Find the closing paren corresponding to the open paren at <index>
       in <string>.  Optionally, can provide other characters to match on.
       If found, returns the index of the matching parenthesis.  If not found,
       returns -1.
    """
    if not string[index] == lparen:
        raise ValueError("Character at index %d is '%s'. Expected '%s'"
                         % (index, string[index], lparen))
    index += 1
    count = 1
    while index < len(string) and count > 0:
        while index < len(string) and string[index] not in (lparen, rparen):
            index += 1
        if string[index] == lparen:
            count += 1
        elif string[index] == rparen:
            count -= 1

    if count == 0:
        return index
    else:
        return -1


def isindex(str):
    """True if a string is something we can index an array with."""
    try:
        int(str)
        return True
    except ValueError:
        return False

def once(function):
    if not hasattr(function, "did_once"):
        function()
        function.did_once = True

# Returns MPI_Blah_[f2c,c2f] prefix for a handle type.  MPI_Datatype is a special case.
def conversion_prefix(handle_type):
    if handle_type == "MPI_Datatype":
        return "MPI_Type"
    else:
        return handle_type

# Special join function for joining lines together.  Puts "\n" at the end too.
def joinlines(list, sep="\n"):
    if list:
        return sep.join(list) + sep
    else:
        return ""

def static_out(function):
    if not function in static_files:
        static_filename = static_dir + '/' + function + '.c'
        file = open(static_filename, "w")
        file.write(wrapper_includes)
        if output_fortran_wrappers:
            file.write(fortran_wrapper_includes)
        if declarations:
            file.write(declarations)
        if output_guards: file.write("extern int in_wrapper;\n")
        static_files[function] = file
    else:
        file = static_files[function]
    return file

# Possible types of Tokens in input.
LBRACE, RBRACE, TEXT, IDENTIFIER = range(4)

class Token:
    """Represents tokens; generated from input by lexer and fed to parse()."""
    def __init__(self, type, value, line=0):
        self.type = type    # Type of token
        self.value = value  # Text value
        self.line = line

    def __str__(self):
        return "'%s'" % re.sub(r'\n', "\\\\n", self.value)

    def isa(self, type):
        return self.type == type


class LineTrackingLexer(object):
    """Base class for Lexers that keep track of line numbers."""
    def __init__(self, lexicon):
        self.line_no = -1
        self.scanner = re.Scanner(lexicon)

    def make_token(self, type, value):
        token = Token(type, value, self.line_no)
        self.line_no += value.count("\n")
        return token

    def lex(self, text):
        self.line_no = 0
        tokens, remainder = self.scanner.scan(text)
        if remainder:
            sys.stderr.write("Unlexable input:\n%s\n" % remainder)
            sys.exit(1)
        self.line_no = -1
        return tokens

class OuterRegionLexer(LineTrackingLexer):
    def __init__(self):
        super(OuterRegionLexer, self).__init__([
            (r'{{',                     self.lbrace),
            (r'}}',                     self.rbrace),
            (r'({(?!{)|}(?!})|[^{}])*', self.text)])
    def lbrace(self, scanner, token): return self.make_token(LBRACE, token)
    def rbrace(self, scanner, token): return self.make_token(RBRACE, token)
    def text(self, scanner, token):   return self.make_token(TEXT, token)

class OuterCommentLexer(OuterRegionLexer):
    def __init__(self):
        super(OuterRegionLexer, self).__init__([
            (r'/\*(.|[\r\n])*?\*/',                self.text),   # multiline comment
            (r'//(.|[\r\n])*?(?=[\r\n])',          self.text),   # single line comment
            (r'{{',                                self.lbrace),
            (r'}}',                                self.rbrace),
            (r'({(?!{)|}(?!})|/(?![/*])|[^{}/])*', self.text)])

class InnerLexer(OuterRegionLexer):
    def __init__(self):
        super(OuterRegionLexer, self).__init__([
            (r'{{',                               self.lbrace),
            (r'}}',                               self.rbrace),
            (r'(["\'])?((?:(?!\1)[^\\]|\\.)*)\1', self.quoted_id),
            (r'([^\s]+)',                         self.identifier),
            (r'\s+', None)])
    def identifier(self, scanner, token): return self.make_token(IDENTIFIER, token)
    def quoted_id(self, scanner, token):
        # remove quotes from quoted ids.  Note that ids and quoted ids are pretty much the same thing;
        # the quotes are just optional.  You only need them if you need spaces in your expression.
        return self.make_token(IDENTIFIER, re.sub(r'^["\'](.*)["\']$', '\\1', token))

# Global current filename and function name for error msgs
cur_filename = ""
cur_function = None

class WrapSyntaxError(Exception):
    """Simple Class for syntax errors raised by the wrapper generator (rather than python)"""
    pass

def syntax_error(msg):
    # TODO: make line numbers actually work.
    sys.stderr.write("%s:%d: %s\n" % (cur_filename, 0, msg))
    if cur_function:
        sys.stderr.write("    While handling %s.\n" % cur_function)
    raise WrapSyntaxError

################################################################################
# MPI Semantics:
#   Classes in this section describe MPI declarations and types.  These are used
#   to parse the mpi.h header and to generate wrapper code.
################################################################################
class Scope:
    """ This is the very basic class for scopes in the wrapper generator.  Scopes
        are hierarchical and support nesting.  They contain string keys mapped
        to either string values or to macro functions.
        Scopes also keep track of the particular macro they correspond to (macro_name).
    """
    def __init__(self, enclosing_scope=None):
        self.map = {}
        self.enclosing_scope = enclosing_scope
        self.macro_name = None           # For better debugging error messages

    def __getitem__(self, key):
        if key in self.map:         return self.map[key]
        elif self.enclosing_scope:  return self.enclosing_scope[key]
        else:                       raise KeyError(key + " is not in scope.")

    def __contains__(self, key):
        if key in self.map:         return True
        elif self.enclosing_scope:  return key in self.enclosing_scope
        else:                       return False

    def __setitem__(self, key, value):
        self.map[key] = value

    def include(self, map):
        """Add entire contents of the map (or scope) to this scope."""
        self.map.update(map)

################################################################################
# MPI Semantics:
#   Classes in this section describe MPI declarations and types.  These are used
#   to parse the mpi.h header and to generate wrapper code.
################################################################################
# Map from function name to declaration created from mpi.h.
mpi_functions = {}

class Param:
    """Descriptor for formal parameters of MPI functions.
       Doesn't represent a full parse, only the initial type information,
       name, and array info of the argument split up into strings.
    """
    def __init__(self, type, pointers, name, array, pos):
        self.type = type               # Name of arg's type (might include things like 'const')
        self.pointers = pointers       # Pointers
        self.name = name               # Formal parameter name (from header or autogenerated)
        self.array = array             # Any array type information after the name
        self.pos = pos                 # Position of arg in declaration
        self.decl = None               # This gets set later by Declaration

    def setDeclaration(self, decl):
        """Needs to be called by Declaration to finish initing the arg."""
        self.decl = decl

    def isArrayIndexOutputParam(self):
        """True if this Param is a pointer to which an array index (or MPI_UNDEFINED) will be written"""
        return (self.decl.name in mpi_array_index_output_calls
                and self.pos == mpi_array_index_output_calls[self.decl.name])

    def isArrayIndexArrayOutputParam(self):
        """True if this Param is an array to which array indices will be written"""
        return (self.decl.name in mpi_array_index_array_output_calls
                and self.pos in mpi_array_index_array_output_calls[self.decl.name])

    def countArrayIndexArrayParam(self):
        """If this Param is an array to which array indices will be written, returns the Param that represents the pointer that is set to the count of its elements"""
        return self.decl.args[mpi_array_index_array_output_calls[self.decl.name][self.pos]]

    def isHandleArray(self):
        """True if this Param represents an array of MPI handle values."""
        return (self.decl.name in mpi_array_calls
                and self.pos in mpi_array_calls[self.decl.name])

    def countParam(self):
        """If this Param is a handle array, returns the Param that represents the count of its elements"""
        return self.decl.args[mpi_array_calls[self.decl.name][self.pos]]

    def isHandle(self):
        """True if this Param is one of the MPI builtin handle types."""
        return self.type in mpi_handle_types

    def isStatus(self):
        """True if this Param is an MPI_Status.  MPI_Status is handled differently
           in c2f/f2c calls from the other handle types.
        """
        return self.type == "MPI_Status"

    def isVoid(self):
        """True if this Param is of type void.
        """
        return self.type == "void"

    def isString(self):
        """True if this Param is of type char*.
        """
        return self.type == "char" and self.pointers

    def isInout(self, function_name):
        """True if this Param is an INOUT argument. c2f and f2c calls
        only for INOUT arguments.
        """
        if self.type == "MPI_File":
            return True
        if re.search("(commit|free|put|set_attr|delete_attr|set_name|_Wait|_Test|_Start)", function_name):
            return True
        return False

    def fortranFormal(self):
        """Prints out a formal parameter for a fortran wrapper."""
        # There are only a few possible fortran arg types in our wrappers, since
        # everything is a pointer.
        if self.type in ("MPI_Aint", "char") or self.type.endswith("_function"):
            ftype = self.type
        else:
            ftype = "MPI_Fint"

        # Arrays don't come in as pointers (they're passed as arrays)
        # Everything else is a pointer.
        if self.pointers:
            pointers = self.pointers
        elif self.array:
            pointers = ""
        else:
            pointers = "*"

        # Put it all together and return the fortran wrapper type here.
        arr = self.array or ''
        return "%s %s%s%s" % (ftype, pointers, self.name, arr)

    def cType(self):
        if not self.type:
            return ''
        else:
            arr = self.array or ''
            pointers = self.pointers or ''
            return "%s%s%s" % (self.type, pointers, arr)

    def cFormal(self):
        """Prints out a formal parameter for a C wrapper."""
        if not self.type:
            return self.name  # special case for '...'
        else:
            arr = self.array or ''
            pointers = self.pointers or ''
            return "%s %s%s%s" % (self.type, pointers, self.name, arr)

    def castType(self):
        arr = self.array or ''
        pointers = self.pointers or ''
        if re.search('\[\s*\]', arr):
            if arr.count('[') > 1:
                pointers += '(*)'   # need extra parens for, e.g., int[][3] -> int(*)[3]
            else:
                pointers += '*'     # justa single array; can pass pointer.
            arr = re.sub('\[\s*\]', '', arr)
        return "%s%s%s" % (self.type, pointers, arr)

    def __str__(self):
        return self.cFormal()


class Declaration:
    """ Descriptor for simple MPI function declarations.
        Contains return type, name of function, and a list of args.
    """
    def __init__(self, rtype, name):
        self.rtype = rtype
        self.name = name
        self.args = []

    def addArgument(self, arg):
        arg.setDeclaration(self)
        self.args.append(arg)

    def __iter__(self):
        for arg in self.args: yield arg

    def __str__(self):
        return self.prototype()

    def retType(self):
        return self.rtype

    def formals(self):
        return [arg.cFormal() for arg in self.args]

    def types(self):
        return [arg.cType() for arg in self.args]

    def argsNoEllipsis(self):
        return filter(lambda arg: arg.name != "...", self.args)

    def returnsErrorCode(self):
        """This is a special case for MPI_Wtime and MPI_Wtick.
           These functions actually return a double value instead of an int error code.
        """
        return self.rtype == "int"

    def hasArrayIndexOutputParam(self):
        """True if this function has an (array) index where MPI_UNDEFINED could be written"""
        return (self.name in mpi_array_index_output_calls or
                self.name in mpi_array_index_array_output_calls)

    def arrayIndexOutputParam(self):
        """If this Param is an array to which array indices will be written, returns the Param that represents the pointer that is set to the count of its elements"""
        if (self.name in mpi_array_index_output_calls):
            return self.args[mpi_array_index_output_calls[self.name]]
        if (self.name in mpi_array_index_array_output_calls):
            return self.args[list(mpi_array_index_array_output_calls[self.name].values())[0]]

    def argNames(self):
        return [arg.name for arg in self.argsNoEllipsis()]

    def getArgName(self, index):
        return self.argsNoEllipsis()[index].name

    def fortranFormals(self):
        formals = list(map(Param.fortranFormal, self.argsNoEllipsis()))
        strLengths = []
        for arg in self.args:
            if arg.isString():
                strLengths.append("int %s_len" % arg.name)

        if self.name == "MPI_Init": formals = []    # Special case for init: no args in fortran
        elif self.name == "MPI_Init_thread": formals = ["MPI_Fint *required", "MPI_Fint *provided"]    # Special case for init: no args in fortran

        ierr = []
        if self.returnsErrorCode(): ierr = ["MPI_Fint *ierr"]
        return formals + ierr + strLengths

    def fortranArgNames(self):
        names = self.argNames()

        strLengths = []
        for arg in self.args:
            if arg.isString():
                strLengths.append("%s_len" % arg.name)

        if self.name == "MPI_Init": names = []
        elif self.name == "MPI_Init_thread": names = ["required", "provided"]

        ierr = []
        if self.returnsErrorCode(): ierr = ["ierr"]
        return names + ierr + strLengths

    def prototype(self, modifiers=""):
        if modifiers: modifiers = joinlines(modifiers, " ")
        return "%s%s %s(%s)" % (modifiers, self.retType(), self.name, ", ".join(self.formals()))

    def pmpi_prototype(self, modifiers=""):
        if modifiers: modifiers = joinlines(modifiers, " ")
        return "%s%s P%s(%s)" % (modifiers, self.retType(), self.name, ", ".join(self.formals()))

    def fortranPrototype(self, name=None, modifiers=""):
        if not name: name = self.name
        if modifiers: modifiers = joinlines(modifiers, " ")

        if self.returnsErrorCode():
            rtype = "void"  # Fortran calls use ierr parameter instead
        else:
            rtype = self.rtype
        return "%s%s %s(%s)" % (modifiers, rtype, name, ", ".join(self.fortranFormals()))


types = set()
all_pointers = set()

def enumerate_mpi_declarations(mpicc, includes):
    """ Invokes mpicc's C preprocessor on a C file that includes mpi.h.
        Parses the output for declarations, and yields each declaration to
        the caller.
    """
    # Create an input file that just includes <mpi.h>
    tmpfile = tempfile.NamedTemporaryFile('w+b', -1, suffix='.c')
    tmpname = "%s" % tmpfile.name
    tmpfile.write(b'#include <mpi.h>')
    tmpfile.write(b"\n")
    tmpfile.flush()

    # Run the mpicc -E on the temp file and pipe the output
    # back to this process for parsing.
    string_includes = ["-I"+dir for dir in includes]
    mpicc_cmd = "%s -E %s" % (mpicc, " ".join(string_includes))
    try:
        popen = subprocess.Popen("%s %s" % (mpicc_cmd, tmpname), shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except IOError:
        sys.stderr.write("IOError: couldn't run '" + mpicc_cmd + "' for parsing mpi.h\n")
        sys.exit(1)

    # Parse out the declarations from the MPI file
    mpi_h = popen.stdout
    for line in mpi_h:
        line = line.decode().strip()
        begin = begin_decl_re.search(line)
        if begin and not exclude_re.search(line):
            # Grab return type and fn name from initial parse
            return_type, fn_name = begin.groups()

            # Accumulate rest of declaration (possibly multi-line)
            while not end_decl_re.search(line):
                line += " " + next(mpi_h).decode().strip()

            # Split args up by commas so we can parse them independently
            fn_and_paren = r'(%s\s*\()' % fn_name
            match = re.search(fn_and_paren, line)
            lparen = match.start(1) + len(match.group(1)) - 1
            rparen = find_matching_paren(line, lparen)
            if rparen < 0:
                raise ValueError("Malformed declaration in header: '%s'" % line)

            arg_string = line[lparen+1:rparen]
            arg_list = list(map(lambda s: s.strip(), arg_string.split(",")))

            # Handle functions that take no args specially
            if arg_list == ['void']:
                arg_list = []

            # Parse formal parameter descriptors out of args
            decl = Declaration(return_type, fn_name)
            arg_num = 0
            for arg in arg_list:
                if arg == '...':   # Special case for Pcontrol.
                    decl.addArgument(Param(None, None, '...', None, arg_num))
                else:
                    match = formal_re.match(arg)
                    if not match:
                        sys.stderr.write("MATCH FAILED FOR: '%s' in %s\n" % (arg, fn_name))
                        sys.exit(1)

                    type, pointers, name, array = match.groups()
                    types.add(type)
                    all_pointers.add(pointers)
                    # If there's no name, make one up.
                    if not name:
                        name = "arg_" + str(arg_num)

                    decl.addArgument(Param(type.strip(), pointers, name, array, arg_num))
                arg_num += 1

            yield decl

    mpi_h.close()
    return_code = popen.wait()
    if return_code != 0:
        sys.stderr.write("Error: Couldn't run '%s' for parsing mpi.h.\n" % mpicc_cmd)
        sys.stderr.write("       Process exited with code %d.\n" % return_code)
        sys.exit(1)

    # Do some cleanup once we're done reading.
    tmpfile.close()


def write_enter_guard(out, decl):
    """Prevent us from entering wrapper functions if we're already in a wrapper function.
       Just call the PMPI function w/o the wrapper instead."""
    if output_guards:
        out.write("    if (in_wrapper) return P%s(%s);\n" % (decl.name, ", ".join(decl.argNames())))
        out.write("    in_wrapper = 1;\n")

def write_exit_guard(out):
    """After a call, set in_wrapper back to 0 so we can enter the next call."""
    if output_guards:
        out.write("    in_wrapper = 0;\n")


def write_c_wrapper(out, decl, return_val, write_body):
    """Write the C wrapper for an MPI function."""
    # Write the PMPI prototype here in case mpi.h doesn't define it
    # (sadly the case with some MPI implementations)
    out.write(decl.pmpi_prototype(default_modifiers))
    out.write(";\n")

    # Now write the wrapper function, which will call the PMPI function we declared.
    out.write(decl.prototype(default_modifiers))
    out.write(" { \n")
    out.write("    %s %s = 0;\n" % (decl.retType(), return_val))

    write_enter_guard(out, decl)
    write_body(out)
    write_exit_guard(out)

    out.write("    return %s;\n" % return_val)
    out.write("}\n\n")


def write_fortran_binding(out, decl, delegate_name, binding, stmts=None):
    """Outputs a wrapper for a particular fortran binding that delegates to the
       primary Fortran wrapper.  Optionally takes a list of statements to execute
       before delegating.
    """
    out.write(decl.fortranPrototype(binding, default_modifiers))
    out.write(" { \n")
    if stmts:
        out.write(joinlines(map(lambda s: "    " + s, stmts)))
    if decl.returnsErrorCode():
        # regular MPI fortran functions use an error code
        out.write("    %s(%s);\n" % (delegate_name, ", ".join(decl.fortranArgNames())))
    else:
        # wtick and wtime return a value
        out.write("    return %s(%s);\n" % (delegate_name, ", ".join(decl.fortranArgNames())))
    out.write("}\n\n")


class FortranDelegation:
    """Class for constructing a call to a Fortran wrapper delegate function.  Provides
       storage for local temporary variables, copies of parameters, callsites for MPI-1 and
       MPI-2, and writebacks to local pointer types.
    """
    def __init__(self, decl, return_val):
        self.decl = decl
        self.return_val = return_val

        self.temps = set()
        self.frees = set()
        self.copies = []
        self.mpich_c2f_copies = []
        self.writebacks = []
        self.mpich_c2f_writebacks = []
        self.actuals = []
        self.mpich_actuals = []
        self.mpich_c2f_actuals = []

    def addTemp(self, type, name):
        """Adds a temp var with a particular name.  Adds the same var only once."""
        temp = "    %s %s;" % (type, name)
        self.temps.add(temp)

    def addFree(self, name):
        """Frees a temp var with a particular name.  Frees the same var only once."""
        temp = " if(%s) free(%s);" % (name, name)
        self.frees.add(temp)

    def addActual(self, actual):
        self.actuals.append(actual)
        self.mpich_actuals.append(actual)
        self.mpich_c2f_actuals.append(actual)

    def addActualMPICH(self, actual):
        self.mpich_actuals.append(actual)

    def addActualMPICH_C2F(self, actual):
        self.mpich_c2f_actuals.append(actual)

    def addActualMPI2(self, actual):
        self.actuals.append(actual)

    def addActualC2F(self, actual):
        self.actuals.append(actual)
        self.mpich_c2f_actuals.append(actual)

    def addWriteback(self, stmt):
        self.writebacks.append("    %s" % stmt)
        self.mpich_c2f_writebacks.append("    %s" % stmt)

    def addCopy(self, stmt):
        self.copies.append("    %s" % stmt)
        self.mpich_c2f_copies.append("    %s" % stmt)

    def addCopyMPI2(self, stmt):
        self.copies.append("    %s" % stmt)

    def addWritebackMPI2(self, stmt):
        self.writebacks.append("    %s" % stmt)

    def addCopyMPICH_C2F(self, stmt):
        self.mpich_c2f_copies.append("    %s" % stmt)

    def addWritebackMPICH_C2F(self, stmt):
        self.mpich_c2f_writebacks.append("    %s" % stmt)

    def write(self, out):
        assert len(self.actuals) == len(self.mpich_actuals)

        call = "    %s = %s" % (self.return_val, self.decl.name)
        mpich_call = "%s(%s);\n" % (call, ", ".join(self.mpich_actuals))
        mpi2_call = "%s(%s);\n" % (call, ", ".join(self.actuals))
        mpich_c2f_call = "%s(%s);\n" % (call, ", ".join(self.mpich_c2f_actuals))

        out.write("    %s %s = 0;\n" % (self.decl.retType(), self.return_val))

        if ignore_deprecated:
            out.write("    WRAP_MPI_CALL_PREFIX\n")

        if mpich_call == mpi2_call and not (self.temps or self.copies or self.writebacks):
            out.write(mpich_call)
        else:
            out.write("#if (!defined(MPICH_HAS_C2F) && defined(MPICH_NAME) && (MPICH_NAME == 1)) /* MPICH test */\n")
            out.write(mpich_call)
            out.write("#else /* MPI-2 safe call */\n")
            out.write(joinlines(self.temps))
            if mpich_c2f_call != mpi2_call:
                out.write("# if defined(MPICH_NAME) && (MPICH_NAME == 1) /* MPICH test */\n")
                out.write(joinlines(self.mpich_c2f_copies))
                out.write(mpich_c2f_call)
                out.write(joinlines(self.mpich_c2f_writebacks))
                out.write("# else /* MPI-2 safe call */\n")
            out.write(joinlines(self.copies))
            out.write(mpi2_call)
            out.write(joinlines(self.writebacks))
            if mpich_c2f_call != mpi2_call:
                out.write("# endif /* MPICH test */\n")
            out.write(joinlines(self.frees))
            out.write("#endif /* MPICH test */\n")

        if ignore_deprecated:
            out.write("    WRAP_MPI_CALL_POSTFIX\n")

def write_fortran_wrappers(out, decl, return_val):
    """Writes primary fortran wrapper that handles arg translation.
       Also outputs bindings for this wrapper for different types of fortran compilers.
    """
    delegate_name = decl.name + f_wrap_suffix

    call = FortranDelegation(decl, return_val)

    if decl.name == "MPI_Init":
        # Use out.write() here so it comes at very beginning of wrapper function
        out.write("    int argc = 0;\n");
        out.write("    char ** argv = NULL;\n");
        call.addActual("&argc");
        call.addActual("&argv");
        if static_dir:
            for binding in pmpi_init_bindings:
                out = static_out(binding)
                out.write(decl.prototype(default_modifiers))
                out.write(";\n");
                out.write(decl.fortranPrototype(delegate_name, ["static"]))
                out.write(" { \n")
                out.write("    int argc = 0;\n");
                out.write("    char ** argv = NULL;\n");
                call.write(out)
                out.write("    *ierr = %s;\n" % return_val)
                out.write("}\n\n")
        else:
            # Use out.write() here so it comes at very beginning of wrapper function
            out.write(decl.fortranPrototype(delegate_name, ["static"]))
            out.write(" { \n")
            out.write("    int argc = 0;\n");
            out.write("    char ** argv = NULL;\n");
            call.write(out)
            out.write("    *ierr = %s;\n" % return_val)
            out.write("}\n\n")

        # Write out various bindings that delegate to the main fortran wrapper
        fortran_init = 1
        for binding in pmpi_init_bindings:
            if static_dir:
                out = static_out(binding)
                out.write("extern int fortran_init;\n")
            write_fortran_binding(out, decl, delegate_name, binding[1:], ["fortran_init = %d;" % fortran_init])
            fortran_init = fortran_init + 1
            if static_dir:
                write_fortran_binding(out, decl, binding, 'real_' + binding)
        return

    elif decl.name == "MPI_Init_thread":
        call.addActual("&argc");
        call.addActual("&argv");
        call.addActual("*required");
        call.addActual("provided");
        if static_dir:
            for binding in pmpi_init_thread_bindings:
                out = static_out(binding)
                out.write(decl.prototype(default_modifiers))
                out.write(";\n");
                out.write(decl.fortranPrototype(delegate_name, ["static"]))
                out.write(" { \n")
                out.write("    int argc = 0;\n");
                out.write("    char ** argv = NULL;\n");
                call.write(out)
                out.write("    *ierr = %s;\n" % return_val)
                out.write("}\n\n")
        else:
            # Use out.write() here so it comes at very beginning of wrapper function
            out.write(decl.fortranPrototype(delegate_name, ["static"]))
            out.write(" { \n")
            out.write("    int argc = 0;\n");
            out.write("    char ** argv = NULL;\n");
            call.write(out)
            out.write("    *ierr = %s;\n" % return_val)
            out.write("}\n\n")

        # Write out various bindings that delegate to the main fortran wrapper
        fortran_init_thread = 1
        for binding in pmpi_init_thread_bindings:
            if static_dir:
                out = static_out(binding)
                out.write("extern int fortran_init_thread;\n")
            write_fortran_binding(out, decl, delegate_name, binding[1:], ["fortran_init_thread = %d;" % fortran_init_thread])
            fortran_init_thread = fortran_init_thread + 1
            if static_dir:
                write_fortran_binding(out, decl, binding, 'real_' + binding)
        return

    else:
        out.write(decl.fortranPrototype(delegate_name, ["static"]))
        out.write(" { \n")

    # For MPI_UNDEFINED we will return instantly, after freeing the temp arrays
    if decl.hasArrayIndexOutputParam():
        call.addWriteback("if (*%s != MPI_UNDEFINED){" % decl.arrayIndexOutputParam().name)

    # This loop processes the rest of the call for all other routines.
    for arg in decl.args:
        if arg.name == "...":   # skip ellipsis
            continue

        if not (arg.pointers or arg.array):
            if not arg.isHandle():
                # These are pass-by-value arguments, so just deref and pass thru
                dereferenced = "*%s" % arg.name
                call.addActual(dereferenced)
            else:
                # Non-ptr, non-arr handles need to be converted with MPI_Blah_f2c
                # No special case for MPI_Status here because MPI_Statuses are never passed by value.
                call.addActualC2F("%s_f2c(*%s)" % (conversion_prefix(arg.type), arg.name))
                call.addActualMPICH("(%s)(*%s)" % (arg.type, arg.name))

        else:
            if (arg.isString()):
                temp = "temp_%s" % arg.name
                call.addActual(arg.name)
                call.addTemp("char*", temp)
                call.addCopy("%s = (%s)malloc(sizeof(%s) * (%s_len+1));" %
                                (temp, "char*", arg.type, arg.name))
                call.addFree(temp)
                call.addCopy("char_p_f2c(%s,%s_len,&%s);"  % (arg.name, arg.name, temp))
                call.addWriteback("char_p_c2f(%s,%s,%s_len);" % (temp, arg.name, arg.name))

            elif (arg.isArrayIndexOutputParam()):
                # Convert from C array index to array Fortran index
                call.addActual("%s" % arg.name)
                call.addWriteback("++(*%s);" % (arg.name))
            elif (arg.isArrayIndexArrayOutputParam()):
                # Convert all C array indices in the array to Fortran array indices
                call.addTemp("int", "i")
                call.addActual("%s" % arg.name)
                call.addWriteback("    for (i=0; i < *%s; ++i)" % arg.countArrayIndexArrayParam().name)
                call.addWriteback("        ++%s[i];" % arg.name)
            elif not arg.isHandle():
                # Non-MPI handle pointer types can be passed w/o dereferencing, but need to
                # cast to correct pointer type first (from MPI_Fint*).
                if arg.isVoid():
                    call.addActual("BufferC2F((%s)%s)" % (arg.castType(), arg.name))
                else:
                    call.addActual("((%s)%s)" % (arg.castType(), arg.name))
            else:
                # For MPI-1, assume ints, cross fingers, and pass things straight through.
                call.addActualMPICH("(%s*)%s" % (arg.type, arg.name))
                conv = conversion_prefix(arg.type)
                temp = "temp_%s" % arg.name

                # For MPI-2, other pointer and array types need temporaries and special conversions.
                if not arg.isHandleArray():
                    call.addTemp(arg.type, temp)

                    if arg.isStatus():
                        call.addActualMPI2("((%s == MPI_F_STATUS_IGNORE) ? MPI_STATUS_IGNORE : &%s)" % (arg.name, temp))
                        call.addActualMPICH_C2F("&%s" % temp)
                        call.addCopyMPI2("if (%s != MPI_F_STATUS_IGNORE) %s_f2c(%s, &%s);"  % (arg.name, conv, arg.name, temp))
                        call.addCopyMPICH_C2F("%s_f2c(%s, &%s);"  % (conv, arg.name, temp))
                        call.addWritebackMPI2("if (%s != MPI_F_STATUS_IGNORE) %s_c2f(&%s, %s);" % (arg.name, conv, temp, arg.name))
                        call.addWritebackMPICH_C2F("%s_c2f(&%s, %s);" % (conv, temp, arg.name))
                    else:
                        call.addActualC2F("&%s" % temp)
                        if arg.isInout(decl.name):
                            call.addCopy("%s = %s_f2c(*%s);"  % (temp, conv, arg.name))
                        call.addWriteback("*%s = %s_c2f(%s);" % (arg.name, conv, temp))
                else:
                    # Make temporary variables for the array and the loop var
                    temp_arr_type = "%s*" % arg.type
                    call.addTemp(temp_arr_type, temp)
                    call.addTemp("int", "i")

                    # Generate the call surrounded by temp array allocation, copies, writebacks, and temp free
                    count = "*%s" % arg.countParam().name
                    call.addCopy("%s = (%s)malloc(sizeof(%s) * %s);" %
                                 (temp, temp_arr_type, arg.type, count))

                    # temp free. This is not included in the !MPI_UNDEFINED block
                    call.addFree(temp)

                    # generate a copy and a writeback statement for this type of handle
                    if arg.isStatus():
                        call.addActualMPI2("((%s == MPI_F_STATUSES_IGNORE) ? MPI_STATUSES_IGNORE : %s)" % (arg.name, temp))
                        call.addActualMPICH_C2F(temp)

                        # Status arrays are OUT args. No convertioning on input.

                        call.addWritebackMPI2("if (%s != MPI_F_STATUSES_IGNORE)" % (arg.name))
                        call.addWriteback("  for (i=0; i < %s; i++)" % count)
                        call.addWritebackMPI2("    %s_c2f(&%s[i], &%s[i * MPI_F_STATUS_SIZE]);" % (conv, temp, arg.name))
                        call.addWritebackMPICH_C2F("    %s_c2f(&%s[i], &%s[i * MPI_F_STATUS_SIZE]);" % (conv, temp, arg.name))
                    else:
                        call.addActualC2F(temp)

                        #if arg.isInout(decl.name):
                        call.addCopy("for (i=0; i < %s; i++)" % count)
                        call.addCopy("    temp_%s[i] = %s_f2c(%s[i]);"  % (arg.name, conv, arg.name))

                        if arg.pointers:
                            call.addWriteback("for (i=0; i < %s; i++)" % count)
                            call.addWriteback("    %s[i] = %s_c2f(temp_%s[i]);" % (arg.name, conv, arg.name))

    if decl.hasArrayIndexOutputParam():
        call.addWriteback("}")

    call.write(out)
    if decl.returnsErrorCode():
        out.write("    *ierr = %s;\n" % return_val)
    else:
        out.write("    return %s;\n" % return_val)
    out.write("}\n\n")

    # Write out various bindings that delegate to the main fortran wrapper
    write_fortran_binding(out, decl, delegate_name, decl.name.upper())
    write_fortran_binding(out, decl, delegate_name, decl.name.lower())
    write_fortran_binding(out, decl, delegate_name, decl.name.lower() + "_")
    write_fortran_binding(out, decl, delegate_name, decl.name.lower() + "__")


################################################################################
# Macros:
#   - functions annotated as @macro or @bodymacro define the global macros and
#     basic pieces of the generator.
#   - include_decl is used to include MPI declarations into function scopes.
################################################################################
# Table of global macros
macros = {}

# This decorator adds macro functions to the outermost function scope.
def macro(macro_name, **attrs):
    def decorate(fun):
        macros[macro_name] = fun # Add macro to outer scope under supplied name
        fun.has_body = False     # By default, macros have no body.
        for key in attrs:        # Optionally set/override attributes
            setattr(fun, key, attrs[key])
        return fun
    return decorate

def handle_list(list_name, list, args):
    """This function handles indexing lists used as macros in the wrapper generator.
       There are two syntaxes:
       {{<list_name>}}          Evaluates to the whole list, e.g. 'foo, bar, baz'
       {{<list_name> <index>}}  Evaluates to a particular element of a list.
    """
    if not args:
        return list
    else:
        len(args) == 1 or syntax_error("Wrong number of args for list expression.")
        try:
            return list[int(args[0])]
        except ValueError:
            syntax_error("Invalid index value: '%s'" % args[0])
        except IndexError:
            syntax_error("Index out of range in '%s': %d" % (list_name, index))

class TypeApplier:
    """This class implements a Macro function for applying something callable to
       args in a decl with a particular type.
    """
    def __init__(self, decl):
        self.decl = decl

    def __call__(self, out, scope, args, children):
        len(args) == 2 or syntax_error("Wrong number of args in apply macro.")
        type, macro_name = args
        for arg in self.decl.args:
            if arg.cType() == type:
                out.write("%s(%s);\n" % (macro_name, arg.name))

class TypeArtTypeApplier:
    """This class implements a Macro function for applying something callable to
       args in a decl with a particular type.
    """
    def __init__(self, decl):
        self.decl = decl

    def _find_count_args(self):
        count_args = []
        count_matcher = re.compile('.*count.*')
        for arg in self.decl.args:
            if count_matcher.match(arg.name) is not None:
                if arg.cType() == 'int':
                    count_args.append((arg, False))
                elif arg.cType() == 'const int[]' or arg.cType() == 'const int*':
                    count_args.append((arg, True))
                else:
                    print("Unknown count argument: ", arg.cType())
        return count_args

    def _find_buffer_args(self):
        buffer_args = []
        buffer_matcher = re.compile('(.*buf.*)|(origin_addr)')
        for arg in self.decl.args:
            if buffer_matcher.match(arg.name) is not None:
                if arg.cType() == 'const void*':
                    buffer_args.append((arg, True))
                elif  arg.cType() == 'void*':
                    buffer_args.append((arg, False))
        return buffer_args

    def _find_type_args(self):
        type_args = []
        type_matcher = re.compile('.*type.*')
        for arg in self.decl.args:
            if arg.cType() == 'MPI_Datatype' and type_matcher.match(arg.name) is not None:
                type_args.append(arg)
        return type_args

    def __call__(self, out, scope, args, children):
        len(args) == 2 or syntax_error("Wrong number of args in apply macro.")
        #type_s, macro_name = args
        #types = type_s.split(',')
        op_type, macro_name = args

        callargs = ['ta_ret_adr']

        buffer_args = self._find_buffer_args()
        count_args = self._find_count_args()
        type_args = self._find_type_args()

        if op_type == 'send':
            if len(buffer_args) != 1:
                raise RuntimeError('%s: Number of buffer args != 1: %d' % (self.decl.name, len(buffer_args)))
            if len(count_args) < 1:
                raise RuntimeError('%s: Number of count args < 1: %d' % (self.decl.name, len(count_args)))
            if len(type_args) < 1:
                raise RuntimeError('%s: Number of type args < 1: %d' % (self.decl.name, len(type_args)))
            callargs += [buffer_args[0][0].name, count_args[0][0].name, type_args[0].name]
        elif op_type == 'recv':
            if len(buffer_args) != 1:
                raise RuntimeError('%s: Number of buffer args != 1: %d' % (self.decl.name, len(buffer_args)))
            if len(count_args) < 1:
                raise RuntimeError('%s: Number of count args < 1: %d' % (self.decl.name, len(count_args)))
            if len(type_args) < 1:
                raise RuntimeError('%s: Number of type args < 1: %d' % (self.decl.name, len(type_args)))
            callargs += [buffer_args[0][0].name, count_args[0][0].name, type_args[0].name]
        elif op_type == 'sendrecv':
            if len(buffer_args) != 2:
                raise RuntimeError('%s: Number of buffer args != 1: %d' % (self.decl.name, len(buffer_args)))
            if len(count_args) < 1:
                raise RuntimeError('%s: Number of count args < 1: %d' % (self.decl.name, len(count_args)))
            if len(type_args) < 1:
                raise RuntimeError('%s: Number of type args < 1: %d' % (self.decl.name, len(type_args)))

            if len(count_args) == 1:
                count_args.append(count_args[0])
            if len(type_args) == 1:
                type_args.append(type_args[0])


            # TODO: Figure out how to properly handle count vectors
            send_count = count_args[0][0].name
            if count_args[0][1]:
                send_count = '1'

            recv_count = count_args[1][0].name
            if count_args[1][1]:
                recv_count = '1'

            callargs += [buffer_args[0][0].name, send_count, type_args[0].name, buffer_args[1][0].name, recv_count, type_args[1].name]
        else:
            return

        arg_string = callargs[0]
        for callarg in callargs[1:]:
            arg_string += ", " + callarg

        out.write("%s(\"%s\", %s);\n" % (macro_name, self.decl.name, arg_string))


'''
        print('  ', macro_name, ': ', self.decl.name)
        pos_buf = 0
        for ty in types:
            pos = 0
            for arg in self.decl.args:
                print(arg)
                pos += 1
                if arg.cType() == ty:
                    argc += 1
                    if callargs == "" and not (arg.name == "count" or arg.name == "sendcount" or arg.name == "recvcount"):
                        pos_buf = pos
                        callargs=arg.name
                    elif (not callargs == "") and pos >= pos_buf and (arg.name == "count" or arg.name == "sendcount" or arg.name == "recvcount"):
                        callargs+=", " + arg.name
                        break
                if argc == len(type):
                    break
        if argc == len(type):
            out.write("%s(\"%s\", %s, %s);\n" % (macro_name, self.decl.name, 'ta_ret_adr', callargs))
            '''

def include_decl(scope, decl):
    """This function is used by macros to include attributes MPI declarations in their scope."""
    scope["ret_type"] = decl.retType()
    scope["args"]     = decl.argNames()
    scope["nargs"]    = len(decl.argNames())
    scope["types"]    = decl.types()
    scope["formals"]  = decl.formals()
    scope["apply_to_type"] = TypeApplier(decl)
    scope["apply_typeart_check"] = TypeArtTypeApplier(decl)
    scope.function_name  = decl.name

    # These are old-stype, deprecated names.
    def get_arg(out, scope, args, children):
        return handle_list("args", decl.argNames(), args)
    scope["get_arg"]     = get_arg
    scope["applyToType"] = scope["apply_to_type"]
    scope["retType"]     = scope["ret_type"]
    scope["argList"]     = "(%s)" % ", ".join(scope["args"])
    scope["argTypeList"] = "(%s)" % ", ".join(scope["formals"])

def all_but(fn_list):
    """Return a list of all mpi functions except those in fn_list"""
    all_mpi = set(mpi_functions.keys())
    diff = all_mpi - set(fn_list)
    return [x for x in sorted(diff)]

@macro("foreachfn", has_body=True)
def foreachfn(out, scope, args, children):
    """Iterate over all functions listed in args."""
    args or syntax_error("Error: foreachfn requires function name argument.")
    global cur_function

    fn_var = args[0]
    for fn_name in args[1:]:
        cur_function = fn_name
        if not fn_name in mpi_functions:
            syntax_error(fn_name + " is not an MPI function")

        fn = mpi_functions[fn_name]
        fn_scope = Scope(scope)
        fn_scope[fn_var] = fn_name
        include_decl(fn_scope, fn)

        for child in children:
            child.evaluate(out, fn_scope)
    cur_function = None

@macro("fn", has_body=True)
def fn(out, scope, args, children):
    """Iterate over listed functions and generate skeleton too."""
    args or syntax_error("Error: fn requires function name argument.")
    global cur_function

    fn_var = args[0]
    for fn_name in args[1:]:
        optional = fn_name.endswith('?')
        if optional:
            fn_name = fn_name[:-1]

        cur_function = fn_name
        if not fn_name in mpi_functions:
            if optional:
                continue
            syntax_error(fn_name + " is not an MPI function")

        fn = mpi_functions[fn_name]
        return_val = "_wrap_py_return_val"

        fn_scope = Scope(scope)
        fn_scope[fn_var] = fn_name
        include_decl(fn_scope, fn)

        fn_scope["ret_val"] = return_val
        fn_scope["returnVal"]  = fn_scope["ret_val"]  # deprecated name.

        if static_dir:
            out = static_out(fn_name)

        if ignore_deprecated:
            c_call = "%s\n%s = P%s(%s);\n%s" % ("WRAP_MPI_CALL_PREFIX", return_val, fn.name, ", ".join(fn.argNames()), "WRAP_MPI_CALL_POSTFIX")
        else:
            c_call = "%s = P%s(%s);" % (return_val, fn.name, ", ".join(fn.argNames()))

        if fn_name == "MPI_Init" and output_fortran_wrappers:
            def callfn(out, scope, args, children):
                # All this is to deal with fortran, since fortran's MPI_Init() function is different
                # from C's.  We need to make sure to delegate specifically to the fortran init wrapping.
                # For dynamic libs, we use weak symbols to pick it automatically.  For static libs, need
                # to rely on input from the user via pmpi_init_binding and the -i option.
                out.write("    if (fortran_init) {\n")
                if static_dir:
                    out.write("        if (!real_PMPI_INIT && !real_pmpi_init && !real_pmpi_init_ && !real_pmpi_init__) {\n")
                    out.write("            fprintf(stderr, \"ERROR: Couldn't find fortran pmpi_init function.  Link against static library instead.\\n\");\n")
                    out.write("            exit(1);\n")
                    out.write("        }")
                    out.write("        switch (fortran_init) {\n")
                    out.write("        case 1: real_PMPI_INIT(&%s);   break;\n" % return_val)
                    out.write("        case 2: real_pmpi_init(&%s);   break;\n" % return_val)
                    out.write("        case 3: real_pmpi_init_(&%s);  break;\n" % return_val)
                    out.write("        case 4: real_pmpi_init__(&%s); break;\n" % return_val)
                    out.write("        default:\n")
                    out.write("            fprintf(stderr, \"NO SUITABLE FORTRAN MPI_INIT BINDING\\n\");\n")
                    out.write("            break;\n")
                    out.write("        }\n")
                else:
                    out.write("#if (defined(PIC) || defined(__PIC__)) && !defined(STATIC)\n")
                    out.write("        if (!PMPI_INIT && !pmpi_init && !pmpi_init_ && !pmpi_init__) {\n")
                    out.write("            fprintf(stderr, \"ERROR: Couldn't find fortran pmpi_init function.  Link against static library instead.\\n\");\n")
                    out.write("            exit(1);\n")
                    out.write("        }")
                    out.write("        switch (fortran_init) {\n")
                    out.write("        case 1: PMPI_INIT(&%s);   break;\n" % return_val)
                    out.write("        case 2: pmpi_init(&%s);   break;\n" % return_val)
                    out.write("        case 3: pmpi_init_(&%s);  break;\n" % return_val)
                    out.write("        case 4: pmpi_init__(&%s); break;\n" % return_val)
                    out.write("        default:\n")
                    out.write("            fprintf(stderr, \"NO SUITABLE FORTRAN MPI_INIT BINDING\\n\");\n")
                    out.write("            break;\n")
                    out.write("        }\n")
                    out.write("#else /* !PIC */\n")
                    out.write("        %s(&%s);\n" % (pmpi_init_binding, return_val))
                    out.write("#endif /* !PIC */\n")
                out.write("    } else {\n")
                out.write("        %s\n" % c_call)
                out.write("    }\n")

            fn_scope["callfn"] = callfn

            def write_fortran_init_flag():
                if static_dir:
                    output.write("int fortran_init = 0;\n")
                    if output_fortran_wrappers:
                        for binding in pmpi_init_bindings:
                            out.write(fn.fortranPrototype("real_%s" % binding, default_modifiers))
                            out.write(";\n");
                            out.write("#pragma weak real_%s\n" % binding);
                else:
                    output.write("static int fortran_init = 0;\n")
            once(write_fortran_init_flag)
            if static_dir:
                out.write("extern int fortran_init;\n")

        elif fn_name == "MPI_Init_thread" and output_fortran_wrappers:
            def callfn(out, scope, args, children):
                # All this is to deal with fortran, since fortran's MPI_Init() function is different
                # from C's.  We need to make sure to delegate specifically to the fortran init wrapping.
                # For dynamic libs, we use weak symbols to pick it automatically.  For static libs, need
                # to rely on input from the user via pmpi_init_binding and the -i option.
                out.write("    if (fortran_init_thread) {\n")
                if static_dir:
                    out.write("        if (!real_PMPI_INIT_THREAD && !real_pmpi_init_thread && !real_pmpi_init_thread_ && !real_pmpi_init_thread__) {\n")
                    out.write("            fprintf(stderr, \"ERROR: Couldn't find fortran pmpi_init_thread function.  Link against static library instead.\\n\");\n")
                    out.write("            exit(1);\n")
                    out.write("        }")
                    out.write("        switch (fortran_init_thread) {\n")
                    out.write("        case 1: real_PMPI_INIT_THREAD(&%s, %s, &%s);   break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        case 2: real_pmpi_init_thread(&%s, %s, &%s);   break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        case 3: real_pmpi_init_thread_(&%s, %s, &%s);  break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        case 4: real_pmpi_init_thread__(&%s, %s, &%s); break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        default:\n")
                    out.write("            fprintf(stderr, \"NO SUITABLE FORTRAN MPI_INIT_THREAD BINDING\\n\");\n")
                    out.write("            break;\n")
                    out.write("        }\n")
                else:
                    out.write("#if (defined(PIC) || defined(__PIC__)) && !defined(STATIC)\n")
                    out.write("        if (!PMPI_INIT_THREAD && !pmpi_init_thread && !pmpi_init_thread_ && !pmpi_init_thread__) {\n")
                    out.write("            fprintf(stderr, \"ERROR: Couldn't find fortran pmpi_init_thread function.  Link against static library instead.\\n\");\n")
                    out.write("            exit(1);\n")
                    out.write("        }")
                    out.write("        switch (fortran_init_thread) {\n")
                    out.write("        case 1: PMPI_INIT_THREAD(&%s, %s, &%s);   break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        case 2: pmpi_init_thread(&%s, %s, &%s);   break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        case 3: pmpi_init_thread_(&%s, %s, &%s);  break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        case 4: pmpi_init_thread__(&%s, %s, &%s); break;\n" % (scope["args"][2], scope["args"][3], return_val))
                    out.write("        default:\n")
                    out.write("            fprintf(stderr, \"NO SUITABLE FORTRAN MPI_INIT_THREAD BINDING\\n\");\n")
                    out.write("            break;\n")
                    out.write("        }\n")
                    out.write("#else /* !PIC */\n")
                    out.write("        %s(&%s, %s, &%s);\n" % (pmpi_init_thread_binding, scope["args"][2], scope["args"][3], return_val))
                    out.write("#endif /* !PIC */\n")
                out.write("    } else {\n")
                out.write("        %s\n" % c_call)
                out.write("    }\n")

            fn_scope["callfn"] = callfn

            def write_fortran_init_thread_flag():
                if static_dir:
                    output.write("int fortran_init_thread = 0;\n")
                    if output_fortran_wrappers:
                        for binding in pmpi_init_thread_bindings:
                            out.write(fn.fortranPrototype("real_%s" % binding, default_modifiers))
                            out.write(";\n");
                            out.write("#pragma weak real_%s\n" % binding);
                else:
                    output.write("static int fortran_init_thread = 0;\n")
            once(write_fortran_init_thread_flag)
            if static_dir:
                out.write("extern int fortran_init_thread;\n")

        else:
            fn_scope["callfn"] = c_call

        def write_body(out):
            for child in children:
                child.evaluate(out, fn_scope)

        out.write("/* ================== C Wrappers for %s ================== */\n" % fn_name)
        write_c_wrapper(out, fn, return_val, write_body)
        if output_fortran_wrappers:
            out.write("/* =============== Fortran Wrappers for %s =============== */\n" % fn_name)
            write_fortran_wrappers(out, fn, return_val)
            out.write("/* ================= End Wrappers for %s ================= */\n\n\n" % fn_name)
    cur_function = None

@macro("forallfn", has_body=True)
def forallfn(out, scope, args, children):
    """Iterate over all but the functions listed in args."""
    args or syntax_error("Error: forallfn requires function name argument.")
    foreachfn(out, scope, [args[0]] + all_but(args[1:]), children)

@macro("fnall", has_body=True)
def fnall(out, scope, args, children):
    """Iterate over all but listed functions and generate skeleton too."""
    args or syntax_error("Error: fnall requires function name argument.")
    fn(out, scope, [args[0]] + all_but(args[1:]), children)

@macro("fnalltype", has_body=True)
def fntype(out, scope, args, children):
    """Iterate over all but listed functions with a specific type and generate skeleton too."""
    len(args) >= 2 or syntax_error("Error: fntype requires function name argument.")
    fn(out, scope, [args[0]] + [decl.name for decl in mpi_functions.values() if args[1] in decl.types() and decl.name not in args[2:]], children)

@macro("sub")
def sub(out, scope, args, children):
    """{{sub <string> <regexp> <substitution>}}
       Replaces value of <string> with all instances of <regexp> replaced with <substitution>.
    """
    len(args) == 3 or syntax_error("'sub' macro takes exactly 4 arguments.")
    string, regex, substitution = args
    if isinstance(string, list):
        return [re.sub(regex, substitution, s) for s in string]
    if not isinstance(regex, str):
        syntax_error("Invalid regular expression in 'sub' macro: '%s'" % regex)
    else:
        return re.sub(regex, substitution, string)

@macro("zip")
def zip_macro(out, scope, args, children):
    len(args) == 2 or syntax_error("'zip' macro takes exactly 2 arguments.")
    if not all([isinstance(a, list) for a in args]):
        syntax_error("Arguments to 'zip' macro must be lists.")
    a, b = args
    return ["%s %s" % x for x in zip(a, b)]

@macro("def")
def def_macro(out, scope, args, children):
    len(args) == 2 or syntax_error("'def' macro takes exactly 2 arguments.")
    scope[args[0]] = args[1]

@macro("list")
def list_macro(out, scope, args, children):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.extend(arg)
        else:
            result.append(arg)
    return result

@macro("filter")
def filter_macro(out, scope, args, children):
    """{{filter <regex> <list>}}
       Returns a list containing all elements of <list> that <regex> matches.
    """
    len(args) == 2 or syntax_error("'filter' macro takes exactly 2 arguments.")
    regex, l = args
    if not isinstance(l, list):
        syntax_error("Invalid list in 'filter' macro: '%s'" % str(list))
    if not isinstance(regex, str):
        syntax_error("Invalid regex in 'filter' macro: '%s'" % str(regex))
    def match(s):
        return re.search(regex, s)
    return list(filter(match, l))

@macro("fn_num")
def fn_num(out, scope, args, children):
    val = fn_num.val
    fn_num.val += 1
    return val
fn_num.val = 0  # init the counter here.

@macro("decls", has_body=True)
def decls(out, scope, args, children):
    global declarations
    declarations = ''
    for child in children:
        declarations += child.text
    out.write(declarations)


################################################################################
# Parser support:
#   - Chunk class for bits of parsed text on which macros are executed.
#   - parse() function uses a Lexer to examine a file.
################################################################################
class Chunk:
    """Represents a piece of a wrapper file.  Is either a text chunk
       or a macro chunk with children to which the macro should be applied.
       macros are evaluated lazily, so the macro is just a string until
       execute is called and it is fetched from its enclosing scope."""
    def __init__(self):
        self.macro    = None
        self.args     = []
        self.text     = None
        self.children = []

    def iwrite(self, file, level, text):
        """Write indented text."""
        for x in xrange(level):
            file.write("  ")
        file.write(text)

    def write(self, file=sys.stdout, l=0):
        if self.macro: self.iwrite(file, l, "{{%s %s}}" % (self.macro, " ".join([str(arg) for arg in self.args])))
        if self.text:  self.iwrite(file, l, "TEXT\n")
        for child in self.children:
            child.write(file, l+1)

    def execute(self, out, scope):
        """This function executes a chunk.  For strings, lists, text chunks, etc., this just
           entails returning the chunk's value.  For callable macros, this executes and returns
           the chunk's value.
        """
        if not self.macro:
            out.write(self.text)
        else:
            if not self.macro in scope:
                error_msg = "Invalid macro: '%s'" % self.macro
                if scope.function_name:
                    error_msg += " for " + scope.function_name
                syntax_error(error_msg)

            value = scope[self.macro]
            if hasattr(value, "__call__"):
                # It's a macro, so we need to execute it.  But first evaluate its args.
                def eval_arg(arg):
                    if isinstance(arg, Chunk):
                        return arg.execute(out, scope)
                    else:
                        return arg
                args = [eval_arg(arg) for arg in self.args]
                return value(out, scope, args, self.children)
            elif isinstance(value, list):
                # Special case for handling lists and list indexing
                return handle_list(self.macro, value, self.args)
            else:
                # Just return the value of anything else
                return value

    def stringify(self, value):
        """Used by evaluate() to print the return values of chunks out to the output file."""
        if isinstance(value, list):
            return ", ".join(value)
        else:
            return str(value)

    def evaluate(self, out, scope):
        """This is an 'interactive' version of execute.  This should be called when
           the chunk's value (if any) should be written out.  Body macros and the outermost
           scope should use this instead of execute().
        """
        value = self.execute(out, scope)
        if value is not None:  # Note the distinction here -- 0 is false but we want to print it!
            out.write(self.stringify(value))

class Parser:
    """Parser for the really simple wrappergen grammar.
       This parser has support for multiple lexers.  self.tokens is a list of iterables, each
       representing a new token stream.  You can add additional tokens to be lexed using push_tokens.
       This will cause the pushed tokens to be handled before any others.  This allows us to switch
       lexers while parsing, so that the outer part of the file is processed in a language-agnostic
       way, but stuff inside macros is handled as its own macro language.
    """
    def __init__(self, macros):
        self.macros = macros
        self.macro_lexer = InnerLexer()
        self.tokens = iter([]) # iterators over tokens, handled in order.  Starts empty.
        self.token = None      # last accepted token
        self.next = None       # next token

    def gettok(self):
        """Puts the next token in the input stream into self.next."""
        try:
            self.next = next(self.tokens)
        except StopIteration:
            self.next = None

    def push_tokens(self, iterable):
        """Adds all tokens in some iterable to the token stream."""
        self.tokens = itertools.chain(iter(iterable), iter([self.next]), self.tokens)
        self.gettok()

    def accept(self, id):
        """Puts the next symbol in self.token if we like it.  Then calls gettok()"""
        if self.next.isa(id):
            self.token = self.next
            self.gettok()
            return True
        return False

    def unexpected_token(self):
        syntax_error("Unexpected token: %s." % self.next)

    def expect(self, id):
        """Like accept(), but fails if we don't like the next token."""
        if self.accept(id):
            return True
        else:
            if self.next:
                self.unexpected_token()
            else:
                syntax_error("Unexpected end of file.")
            sys.exit(1)

    def is_body_macro(self, name):
        """Shorthand for testing whether a particular name is the name of a macro that has a body.
           Need this for parsing the language b/c things like {{fn}} need a corresponding {{endfn}}.
        """
        return name in self.macros and self.macros[name].has_body

    def macro(self, accept_body_macros=True):
        # lex inner-macro text as wrapper language if we encounter text here.
        if self.accept(TEXT):
            self.push_tokens(self.macro_lexer.lex(self.token.value))

        # Now proceed with parsing the macro language's tokens
        chunk = Chunk()
        self.expect(IDENTIFIER)
        chunk.macro = self.token.value

        if not accept_body_macros and self.is_body_macro(chunk.macro):
            syntax_error("Cannot use body macros in expression context: '%s'" % chunk.macro)
            eys.exit(1)

        while True:
            if self.accept(LBRACE):
                chunk.args.append(self.macro(False))
            elif self.accept(IDENTIFIER):
                chunk.args.append(self.token.value)
            elif self.accept(TEXT):
                self.push_tokens(self.macro_lexer.lex(self.token.value))
            else:
                self.expect(RBRACE)
                break
        return chunk

    def text(self, end_macro = None):
        chunks = []
        while self.next:
            if self.accept(TEXT):
                chunk = Chunk()
                chunk.text = self.token.value
                chunks.append(chunk)
            elif self.accept(LBRACE):
                chunk = self.macro()
                name = chunk.macro

                if name == end_macro:
                    # end macro: just break and don't append
                    break
                elif isindex(chunk.macro):
                    # Special case for indices -- raw number macros index 'args' list
                    chunk.macro = "args"
                    chunk.args = [name]
                elif self.is_body_macro(name):
                    chunk.children = self.text("end"+name)
                chunks.append(chunk)
            else:
                self.unexpected_token()

        return chunks

    def parse(self, text):
        if skip_headers:
            outer_lexer = OuterRegionLexer()   # Not generating C code, text is text.
        else:
            outer_lexer = OuterCommentLexer()  # C code. Considers C-style comments.
        self.push_tokens(outer_lexer.lex(text))
        return self.text()

################################################################################
# Main script:
#   Get arguments, set up outer scope, parse files, generator wrappers.
################################################################################
def usage():
    sys.stderr.write(usage_string)
    sys.exit(2)

# Let the user specify another mpicc to get mpi.h from
output = sys.stdout
output_filename = None
static_files = dict()

try:
    opts, args = getopt.gnu_getopt(sys.argv[1:], "fsgdwc:o:i:I:S:")
except getopt.GetoptError as err:
    sys.stderr.write(str(err) + "\n")
    usage()

for opt, arg in opts:
    if opt == "-d": dump_prototypes = True
    if opt == "-f": output_fortran_wrappers = True
    if opt == "-s": skip_headers = True
    if opt == "-g": output_guards = True
    if opt == "-w": ignore_deprecated = True
    if opt == "-c": mpicc = arg
    if opt == "-I":
        stripped = arg.strip()
        if stripped: includes.append(stripped)
    if opt == "-i":
        if not arg in pmpi_init_bindings:
            sys.stderr.write("ERROR: PMPI_Init binding must be one of:\n    %s\n" % " ".join(possible_bindings))
            usage()
        else:
            pmpi_init_binding = arg
            pmpi_init_thread_binding = arg.replace('mpi_init', 'mpi_init_thread').replace('MPI_INIT', 'MPI_INIT_THREAD')
    if opt == "-o":
        try:
            output_filename = arg
            output = open(output_filename, "w")
        except IOError:
            sys.stderr.write("Error: couldn't open file " + arg + " for writing.\n")
            sys.exit(1)
    if opt == "-S":
        static_dir = arg

if static_dir:
    try:
        if not output_filename:
            output_filename = static_dir + '/wrap.c'
            output = open(output_filename, "w")
    except IOError:
        sys.stderr.write("Error: couldn't open file " + arg + " for writing.\n")
        sys.exit(1)

if len(args) < 1 and not dump_prototypes:
    usage()

# Parse mpi.h and put declarations into a map.
#
for decl in enumerate_mpi_declarations(mpicc,includes):
    mpi_functions[decl.name] = decl
    if dump_prototypes: print(decl)

# Fail gracefully if we didn't find anything.
if not mpi_functions:
    sys.stderr.write("Error: Found no declarations in mpi.h.\n")
    sys.exit(1)

# If we're just dumping prototypes, we can just exit here.
if dump_prototypes: sys.exit(0)

try:
    # Start with some headers and definitions.
    if not skip_headers:
        output.write(wrapper_includes)
        if output_guards:
            if static_dir:
                output.write("int in_wrapper = 0;\n")
            else:
                output.write("static int in_wrapper = 0;\n")
        if output_fortran_wrappers:
            output.write(fortran_wrapper_includes)
        if not static_dir:
            output.write(wrapper_main_pmpi_init_decls)

    # Print the macros for disabling MPI function deprecation warnings.
    if ignore_deprecated:
        output.write(wrapper_diagnosics_macros)

    # Parse each file listed on the command line and execute
    # it once it's parsed.
    fileno = 0
    for f in args:
        cur_filename = f
        file = open(cur_filename)

        # Outer scope contains fileno and the fundamental macros.
        outer_scope = Scope()
        outer_scope["fileno"] = str(fileno)
        outer_scope.include(macros)

        parser = Parser(macros)
        chunks = parser.parse(file.read())

        for chunk in chunks:
            chunk.evaluate(output, Scope(outer_scope))
        fileno += 1

except WrapSyntaxError:
    output.close()
    for file in static_files.values():
        file.close()
    if output_filename: os.remove(output_filename)
    sys.exit(1)

output.close()
for file in static_files.values():
    file.close()
