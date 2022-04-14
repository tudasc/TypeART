// This file tests for an specific endless recursion in the filter implementations w.r.t. following store targets
// RUN: %c-to-llvm -fno-discard-value-names %s | %opt -O3 -S \
// RUN: | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 \
// RUN: | %filecheck %s --check-prefix=CHECK-exp-default-opt

// CHECK-exp-default-opt: TypeArtPass [Heap & Stack]
// CHECK-exp-default-opt-next: Malloc :   1
// CHECK-exp-default-opt-next: Free   :   1
// CHECK-exp-default-opt-next: Alloca :   2
// CHECK-exp-default-opt-next: Global :   0

#include <stdlib.h>

#define hypre_TAlloc(type, count) \
  ((unsigned int)(count) * sizeof(type)) > 0 ? ((type*)malloc((unsigned int)(sizeof(type) * (count)))) : (type*)NULL

#define hypre_CTAlloc(type, count)                                                                                \
  ((unsigned int)(count) * sizeof(type)) > 0 ? ((type*)calloc((unsigned int)(count), (unsigned int)sizeof(type))) \
                                             : (type*)NULL

#define hypre_TFree(ptr) (free((char*)ptr), ptr = NULL)

typedef int hypre_Index[3];
typedef int* hypre_IndexRef;

typedef struct hypre_Box_struct {
  hypre_Index imin; /* min bounding indices */
  hypre_Index imax; /* max bounding indices */

} hypre_Box;

typedef struct hypre_BoxArray_struct {
  hypre_Box* boxes; /* Array of boxes */
  int size;         /* Size of box array */
  int alloc_size;   /* Size of currently allocated space */

} hypre_BoxArray;

#define hypre_BoxArrayExcess 10

typedef struct hypre_BoxArrayArray_struct {
  hypre_BoxArray** box_arrays; /* Array of pointers to box arrays */
  int size;                    /* Size of box array array */

} hypre_BoxArrayArray;

#define hypre_IndexD(index, d) (index[d])

#define hypre_IndexX(index) hypre_IndexD(index, 0)
#define hypre_IndexY(index) hypre_IndexD(index, 1)
#define hypre_IndexZ(index) hypre_IndexD(index, 2)

#define hypre_SetIndex(index, ix, iy, iz) (hypre_IndexX(index) = ix, hypre_IndexY(index) = iy, hypre_IndexZ(index) = iz)

#define hypre_ClearIndex(index) hypre_SetIndex(index, 0, 0, 0)

#define hypre_BoxIMin(box) ((box)->imin)
#define hypre_BoxIMax(box) ((box)->imax)

#define hypre_BoxOffsetDistance(box, index) \
  (hypre_IndexX(index) + (hypre_IndexY(index) + (hypre_IndexZ(index) * hypre_BoxSizeY(box))) * hypre_BoxSizeX(box))

#define hypre_CCBoxOffsetDistance(box, index) 0

#define hypre_BoxArrayBoxes(box_array)     ((box_array)->boxes)
#define hypre_BoxArrayBox(box_array, i)    &((box_array)->boxes[(i)])
#define hypre_BoxArraySize(box_array)      ((box_array)->size)
#define hypre_BoxArrayAllocSize(box_array) ((box_array)->alloc_size)

#define hypre_BoxArrayArrayBoxArrays(box_array_array)   ((box_array_array)->box_arrays)
#define hypre_BoxArrayArrayBoxArray(box_array_array, i) ((box_array_array)->box_arrays[(i)])
#define hypre_BoxArrayArraySize(box_array_array)        ((box_array_array)->size)

extern int hypre_UnionBoxes(hypre_BoxArray* boxes);
extern hypre_BoxArrayArray* hypre_BoxArrayArrayCreate(int);
extern void hypre_AppendBox(hypre_Box*, hypre_BoxArray*);
extern void hypre_BoxSetExtents(hypre_Box*, hypre_Index, hypre_Index);
extern void hypre_BoxArrayArrayDestroy(hypre_BoxArrayArray*);

int hypre_MinUnionBoxes(hypre_BoxArray* boxes) {
  int ierr = 0;

  hypre_BoxArrayArray* rotated_array;
  hypre_BoxArray* rotated_boxes;
  hypre_Box *box, *rotated_box;
  hypre_Index lower, upper;

  int i, j, size, min_size, array;

  size          = hypre_BoxArraySize(boxes);
  rotated_box   = hypre_CTAlloc(hypre_Box, 1);
  rotated_array = hypre_BoxArrayArrayCreate(5);

  for (i = 0; i < 5; i++) {
    rotated_boxes = hypre_BoxArrayArrayBoxArray(rotated_array, i);
    switch (i) {
      case 0:
        for (j = 0; j < size; j++) {
          box = hypre_BoxArrayBox(boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(box)[0], hypre_BoxIMin(box)[2], hypre_BoxIMin(box)[1]);
          hypre_SetIndex(upper, hypre_BoxIMax(box)[0], hypre_BoxIMax(box)[2], hypre_BoxIMax(box)[1]);
          hypre_BoxSetExtents(rotated_box, lower, upper);
          hypre_AppendBox(rotated_box, rotated_boxes);
        }
        hypre_UnionBoxes(rotated_boxes);
        break;

      case 1:
        for (j = 0; j < size; j++) {
          box = hypre_BoxArrayBox(boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(box)[1], hypre_BoxIMin(box)[2], hypre_BoxIMin(box)[0]);
          hypre_SetIndex(upper, hypre_BoxIMax(box)[1], hypre_BoxIMax(box)[2], hypre_BoxIMax(box)[0]);
          hypre_BoxSetExtents(rotated_box, lower, upper);
          hypre_AppendBox(rotated_box, rotated_boxes);
        }
        hypre_UnionBoxes(rotated_boxes);
        break;

      case 2:
        for (j = 0; j < size; j++) {
          box = hypre_BoxArrayBox(boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(box)[1], hypre_BoxIMin(box)[0], hypre_BoxIMin(box)[2]);
          hypre_SetIndex(upper, hypre_BoxIMax(box)[1], hypre_BoxIMax(box)[0], hypre_BoxIMax(box)[2]);
          hypre_BoxSetExtents(rotated_box, lower, upper);
          hypre_AppendBox(rotated_box, rotated_boxes);
        }
        hypre_UnionBoxes(rotated_boxes);
        break;

      case 3:
        for (j = 0; j < size; j++) {
          box = hypre_BoxArrayBox(boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(box)[2], hypre_BoxIMin(box)[0], hypre_BoxIMin(box)[1]);
          hypre_SetIndex(upper, hypre_BoxIMax(box)[2], hypre_BoxIMax(box)[0], hypre_BoxIMax(box)[1]);
          hypre_BoxSetExtents(rotated_box, lower, upper);
          hypre_AppendBox(rotated_box, rotated_boxes);
        }
        hypre_UnionBoxes(rotated_boxes);
        break;

      case 4:
        for (j = 0; j < size; j++) {
          box = hypre_BoxArrayBox(boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(box)[2], hypre_BoxIMin(box)[1], hypre_BoxIMin(box)[0]);
          hypre_SetIndex(upper, hypre_BoxIMax(box)[2], hypre_BoxIMax(box)[1], hypre_BoxIMax(box)[0]);
          hypre_BoxSetExtents(rotated_box, lower, upper);
          hypre_AppendBox(rotated_box, rotated_boxes);
        }
        hypre_UnionBoxes(rotated_boxes);
        break;

    } /*switch(i) */
  }   /* for (i= 0; i< 5; i++) */
  hypre_TFree(rotated_box);

  hypre_UnionBoxes(boxes);

  array    = 5;
  min_size = hypre_BoxArraySize(boxes);

  for (i = 0; i < 5; i++) {
    rotated_boxes = hypre_BoxArrayArrayBoxArray(rotated_array, i);
    if (hypre_BoxArraySize(rotated_boxes) < min_size) {
      min_size = hypre_BoxArraySize(rotated_boxes);
      array    = i;
    }
  }

  /* copy the box_array with the minimum number of boxes to boxes */
  if (array != 5) {
    rotated_boxes             = hypre_BoxArrayArrayBoxArray(rotated_array, array);
    hypre_BoxArraySize(boxes) = min_size;

    switch (array) {
      case 0:
        for (j = 0; j < min_size; j++) {
          rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(rotated_box)[0], hypre_BoxIMin(rotated_box)[2],
                         hypre_BoxIMin(rotated_box)[1]);
          hypre_SetIndex(upper, hypre_BoxIMax(rotated_box)[0], hypre_BoxIMax(rotated_box)[2],
                         hypre_BoxIMax(rotated_box)[1]);

          hypre_BoxSetExtents(hypre_BoxArrayBox(boxes, j), lower, upper);
        }
        break;

      case 1:
        for (j = 0; j < min_size; j++) {
          rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(rotated_box)[2], hypre_BoxIMin(rotated_box)[0],
                         hypre_BoxIMin(rotated_box)[1]);
          hypre_SetIndex(upper, hypre_BoxIMax(rotated_box)[2], hypre_BoxIMax(rotated_box)[0],
                         hypre_BoxIMax(rotated_box)[1]);

          hypre_BoxSetExtents(hypre_BoxArrayBox(boxes, j), lower, upper);
        }
        break;

      case 2:
        for (j = 0; j < min_size; j++) {
          rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(rotated_box)[1], hypre_BoxIMin(rotated_box)[0],
                         hypre_BoxIMin(rotated_box)[2]);
          hypre_SetIndex(upper, hypre_BoxIMax(rotated_box)[1], hypre_BoxIMax(rotated_box)[0],
                         hypre_BoxIMax(rotated_box)[2]);

          hypre_BoxSetExtents(hypre_BoxArrayBox(boxes, j), lower, upper);
        }
        break;

      case 3:
        for (j = 0; j < min_size; j++) {
          rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(rotated_box)[1], hypre_BoxIMin(rotated_box)[2],
                         hypre_BoxIMin(rotated_box)[0]);
          hypre_SetIndex(upper, hypre_BoxIMax(rotated_box)[1], hypre_BoxIMax(rotated_box)[2],
                         hypre_BoxIMax(rotated_box)[0]);

          hypre_BoxSetExtents(hypre_BoxArrayBox(boxes, j), lower, upper);
        }
        break;

      case 4:
        for (j = 0; j < min_size; j++) {
          rotated_box = hypre_BoxArrayBox(rotated_boxes, j);
          hypre_SetIndex(lower, hypre_BoxIMin(rotated_box)[2], hypre_BoxIMin(rotated_box)[1],
                         hypre_BoxIMin(rotated_box)[0]);
          hypre_SetIndex(upper, hypre_BoxIMax(rotated_box)[2], hypre_BoxIMax(rotated_box)[1],
                         hypre_BoxIMax(rotated_box)[0]);

          hypre_BoxSetExtents(hypre_BoxArrayBox(boxes, j), lower, upper);
        }
        break;

    } /* switch(array) */
  }   /* if (array != 5) */

  hypre_BoxArrayArrayDestroy(rotated_array);

  return ierr;
}