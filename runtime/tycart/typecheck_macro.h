//
// Created by mority on 12/7/19.
//

#ifndef TYPECHECK_MACRO_TYPECHECK_MACRO_H
#define TYPECHECK_MACRO_TYPECHECK_MACRO_H

// void __typeart_assert_type_stub(void * ptr, void * __type_ptr) {}
// void __typeart_assert_type_stub_len(void * ptr, void * __type_ptr, long len) {}
//
//#define ASSERT_TYPE(ptr, type)                                  \
//  {                                                             \
//    type* __type_ptr;                                           \
//    __typeart_assert_type_stub(ptr, __type_ptr);                \
//  }
//
//#define ASSERT_TYPE(ptr, type, len)                             \
//  {                                                             \
//    type* __type_ptr;                                           \
//    __typeart_assert_type_stub_len(ptr, __type_ptr, len);       \
//  }

#include "RuntimeInterface.h"

#ifdef __VELOC_H

#define TY_protect(id, pointer, count, type)             \
  {                                                      \
    ASSERT_TYPE(pointer, type, count);                   \
    VELOC_Mem_protect(id, pointer, count, sizeof(type)); \
  }

#else

#ifdef __FTI_H__

// define getter functions for FTI's basic types

// char ---------------------------
FTIT_type get_FTI_char() {
  return FTI_CHAR;
}

FTIT_type get_FTI_signed_char() {
  return FTI_CHAR;
}

FTIT_type get_FTI_char_signed() {
  return FTI_CHAR;
}
//---------------------------------

// short --------------------------
FTIT_type get_FTI_short() {
  return FTI_SHRT;
}

FTIT_type get_FTI_signed_short() {
  return FTI_SHRT;
}

FTIT_type get_FTI_short_signed() {
  return FTI_SHRT;
}

FTIT_type get_FTI_short_int() {
  return FTI_SHRT;
}

FTIT_type get_FTI_signed_short_int() {
  return FTI_SHRT;
}

FTIT_type get_FTI_short_signed_int() {
  return FTI_SHRT;
}

FTIT_type get_FTI_short_int_signed() {
  return FTI_SHRT;
}

FTIT_type get_FTI_int_short() {
  return FTI_SHRT;
}

FTIT_type get_FTI_signed_int_short() {
  return FTI_SHRT;
}

FTIT_type get_FTI_int_signed_short() {
  return FTI_SHRT;
}

FTIT_type get_FTI_int_short_signed() {
  return FTI_SHRT;
}
//---------------------------------

// int ----------------------------
FTIT_type get_FTI_int() {
  return FTI_INTG;
}

FTIT_type get_FTI_signed_int() {
  return FTI_INTG;
}

FTIT_type get_FTI_int_signed() {
  return FTI_INTG;
}
//---------------------------------

// long ---------------------------
FTIT_type get_FTI_long() {
  return FTI_LONG;
}

FTIT_type get_FTI_signed_long() {
  return FTI_LONG;
}

FTIT_type get_FTI_long_signed() {
  return FTI_LONG;
}

FTIT_type get_FTI_long_int() {
  return FTI_LONG;
}

FTIT_type get_FTI_signed_long_int() {
  return FTI_LONG;
}

FTIT_type get_FTI_long_signed_int() {
  return FTI_LONG;
}

FTIT_type get_FTI_long_int_signed() {
  return FTI_LONG;
}

FTIT_type get_FTI_int_long() {
  return FTI_LONG;
}

FTIT_type get_FTI_signed_int_long() {
  return FTI_LONG;
}

FTIT_type get_FTI_int_signed_long() {
  return FTI_LONG;
}

FTIT_type get_FTI_int_long_signed() {
  return FTI_LONG;
}
//---------------------------------

// unsigned char ------------------
FTIT_type get_FTI_unsigned_char() {
  return FTI_UCHR;
}

FTIT_type get_FTI_char_unsigned() {
  return FTI_UCHR;
}
//---------------------------------

// unsigned short -----------------
FTIT_type get_FTI_unsigned_short() {
  return FTI_USHT;
}

FTIT_type get_FTI_int_unsigned_short() {
  return FTI_USHT;
}

FTIT_type get_FTI_unsigned_int_short() {
  return FTI_USHT;
}

FTIT_type get_FTI_unsigned_short_int() {
  return FTI_USHT;
}

FTIT_type get_FTI_short_unsigned() {
  return FTI_USHT;
}

FTIT_type get_FTI_int_short_unsigned() {
  return FTI_USHT;
}

FTIT_type get_FTI_short_int_unsigned() {
  return FTI_USHT;
}

FTIT_type get_FTI_short_unsigned_int() {
  return FTI_USHT;
}
//---------------------------------

// unsigned int -------------------
FTIT_type get_FTI_unsigned_int() {
  return FTI_UINT;
}

FTIT_type get_FTI_int_unsigned() {
  return FTI_UINT;
}
//---------------------------------

// unsigned long ------------------
FTIT_type get_FTI_unsigned_long() {
  return FTI_ULNG;
}

FTIT_type get_FTI_int_unsigned_long() {
  return FTI_ULNG;
}

FTIT_type get_FTI_unsigned_int_long() {
  return FTI_ULNG;
}

FTIT_type get_FTI_unsigned_long_int() {
  return FTI_ULNG;
}

FTIT_type get_FTI_long_unsigned() {
  return FTI_ULNG;
}

FTIT_type get_FTI_int_long_unsigned() {
  return FTI_ULNG;
}

FTIT_type get_FTI_long_int_unsigned() {
  return FTI_ULNG;
}

FTIT_type get_FTI_long_unsigned_int() {
  return FTI_ULNG;
}

// float --------------------------
FTIT_type get_FTI_float() {
  return FTI_SFLT;
}
//---------------------------------

// double -------------------------
FTIT_type get_FTI_double() {
  return FTI_DBLE;
}
//---------------------------------

// long double --------------------
FTIT_type get_FTI_long_double() {
  return FTI_LDBE;
}

FTIT_type get_FTI_double_long() {
  return FTI_LDBE;
}
//---------------------------------

#define GET_MACRO(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define TY_protect(...) GET_MACRO(__VA_ARGS__, TY_protect6, TY_protect5, TY_protect4)(__VA_ARGS__)

#define TY_protect4(id, pointer, count, type)          \
  {                                                    \
    ASSERT_TYPE(pointer, type, count);                 \
    FTI_Protect(id, pointer, count, get_FTI_##type()); \
  }

#define TY_protect5(id, pointer, count, type1, type2)             \
  {                                                               \
    ASSERT_TYPE(pointer, type1 type2, count);                     \
    FTI_Protect(id, pointer, count, get_FTI_##type1##_##type2()); \
  }

#define TY_protect6(id, pointer, count, type1, type2, type3)                \
  {                                                                         \
    ASSERT_TYPE(pointer, type1 type2 type3, count);                         \
    FTI_Protect(id, pointer, count, get_FTI_##type1##_##type2##_##type3()); \
  }

#else

#ifdef MINI_CPR_MINI_CPR_H

#define TY_protect(id, pointer, count, type)             \
  {                                                      \
    ASSERT_TYPE(pointer, type, count);                   \
    mini_cpr_register(id, pointer, count, sizeof(type)); \
  }

#endif  // MINI_CPR_MINI_CPR_H
#endif  //__FTI_H__
#endif  //__VELOC_H
#endif  // TYPECHECK_MACRO_TYPECHECK_MACRO_H
