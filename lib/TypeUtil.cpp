#include "TypeUtil.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instructions.h"


#include <iostream>

using namespace llvm;

namespace util {
	namespace type {

		/**
		 * Code was imported from jplehr/llvm-memprofiler project
		 */
		int getTypeSizeInBytes(llvm::Type *t, const llvm::DataLayout &dl){
			int bytes = getScalarSizeInBytes(t);

			if (t->isArrayTy()) {
				bytes = getArraySizeInBytes(t, dl);
			} else if (t->isStructTy()) {
				bytes = getStructSizeInBytes(t, dl);
			} else if (t->isPointerTy()) {
				bytes = getPointerSizeInBytes(t, dl);
			}

			return bytes;
		}

		int getScalarSizeInBytes(llvm::Type *t){
			return t->getScalarSizeInBits() / 8; 	
		}

		int getArraySizeInBytes(llvm::Type *arrT, const llvm::DataLayout &dl){
			auto st = dyn_cast<ArrayType>(arrT);
			Type *underlyingType = st->getElementType();
			int bytes = getScalarSizeInBytes(underlyingType);
			bytes *= st->getNumElements();
			std::cout << "Determined number of bytes to allocate: " << bytes << std::endl;

			return bytes;
		}

		int getStructSizeInBytes(llvm::Type *structT, const llvm::DataLayout &dl){
			int bytes = 0;
			for (auto it = structT->subtype_begin(); it != structT->subtype_end(); ++it) {
				bytes += getTypeSizeInBytes(*it, dl);
			}
			return bytes;
		}

		int getPointerSizeInBytes(llvm::Type *ptrT, const llvm::DataLayout &dl){
			return dl.getPointerSizeInBits() / 8;
		}

		int getTypeSizeForArrayAlloc(llvm::AllocaInst *ai, const llvm::DataLayout &dl){
			int bytes = ai->getAllocatedType()->getScalarSizeInBits() / 8;
			if (ai->isArrayAllocation()) {
				if (auto ci = dyn_cast<ConstantInt>(ai->getArraySize())) {
					bytes *= ci->getLimitedValue();
				} else {
					// If this can not be determined statically, we have to compute it at
					// runtime. We insert additional instructions to calculate the
					// numBytes of that array on the fly. (VLAs produce this kind of
					// behavior)
					// ATTENTION: We can have multiple such arrays in a single BB. We need
					// to have a small vector to store whether we already generated
					// instructions, to possibly refer to the results for further
					// calculations.
					std::cout << "We hit not yet determinable array size expression\n";
				}
			}
			return bytes;
		}

	} // type
} // util 
