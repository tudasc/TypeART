//
// Created by mority on 2/10/20.
//

#ifndef TYPECHECK_MACRO_GOL_TESTING_H
#define TYPECHECK_MACRO_GOL_TESTING_H



// macro to set actual type
#ifndef ACTUAL_TYPE
#define ACTUAL_TYPE char
#endif

// macro to set the actual type when testing the fti version
#ifndef ACTUAL_TYPE_FTI
#define ACTUAL_TYPE_FTI char
#endif

// macro to set expected type
#ifndef EXPECTED_TYPE
#define EXPECTED_TYPE char
#endif

// macro to set the expected type when testing the fti version
#ifndef EXPECTED_TYPE_FTI
#define EXPECTED_TYPE_FTI char
#endif

// macro to set actual X dimension
#ifndef ACTUAL_X
#define ACTUAL_X 50
#endif

// macro to set actual Y dimension
#ifndef ACTUAL_Y
#define ACTUAL_Y 50
#endif

// macro for actual size
#define ACTUAL_SIZE ACTUAL_X * ACTUAL_Y

// macro to set expected X dimension
#ifndef EXPECTED_X
#define EXPECTED_X 50
#endif

// macro to set expected Y dimension
#ifndef EXPECTED_Y
#define EXPECTED_Y 50
#endif

// macro for expected size
#ifndef EXPECTED_SIZE
#define EXPECTED_SIZE EXPECTED_X * EXPECTED_Y
#endif

#endif //TYPECHECK_MACRO_GOL_TESTING_H
