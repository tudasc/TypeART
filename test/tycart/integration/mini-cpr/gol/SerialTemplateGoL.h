#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>

#ifdef PAPI_MEASUREMENT
	#include "PapiInstance.h"
#endif

namespace util{
	static int getIdx(const int i, const int j, const int dimX){
		return j * dimX + i;
	}
}

struct InitFunction {
	int operator()(const int i, const int j){
		const int v = (j-i > 0)?j-i:1;
		int r = (i+j) / (1 + v);
		return (r == 0) ? i+1 : r;
	}
};

template<typename DType>
class GoLStencil {
	public:
		GoLStencil(int dimX, int dimY) : dimX(dimX), dimY(dimY) {};

		void apply(const int i, const int j, const std::vector<DType> &grid, std::vector<DType> &newGrid);

	private:
		int getNumLiveNeighbors(const int i, const int j, const std::vector<DType> &grid);
		int dimX, dimY;
};


template<typename DType>
int GoLStencil<DType>::getNumLiveNeighbors(const int i, const int j, const std::vector<DType> &grid){
	int neighbors = 0;
	
	int idx = util::getIdx(i, j, dimX);
	
	int w = util::getIdx(i, j-1, dimX);
	int e = util::getIdx(i, j+1, dimX);


	int n = util::getIdx(i-1, j, dimX);
	int ne = util::getIdx(i-1, j+1, dimX);
	int nw = util::getIdx(i-1, j-1, dimX);

	int se = util::getIdx(i+1, j+1, dimX);
	int s = util::getIdx(i+1, j, dimX);
	int sw = util::getIdx(i+1, j-1, dimX);
	
	auto neighborElems = {n, ne, e, se, s , sw, w, nw};

	for(const auto &neighbor : neighborElems){
		if(neighbor >= 0 && neighbor < grid.size()){
			if(grid.at(neighbor) == 'l'){
				neighbors++;
			}
		}
	}

	return neighbors;
}


template<typename DType>
void GoLStencil<DType>::apply(const int i, const int j, const std::vector<DType> &grid, std::vector<DType> &newGrid){
	int numLiveNeighbors = getNumLiveNeighbors(i, j, grid);

	int idx = util::getIdx(i, j, dimX);

	if(grid.at(idx) == 'l' && numLiveNeighbors < 2){
		newGrid.at(idx) = 'd';
		return;
	}

	if(grid.at(idx) == 'l'){
		if(numLiveNeighbors == 2 || numLiveNeighbors == 3){
			newGrid.at(idx) = 'l';
			return;
		}
	}

	if(grid.at(idx) == 'l' && numLiveNeighbors > 3){
		newGrid.at(idx) = 'd';
		return;
	}

	if(grid.at(idx) == 'd' && numLiveNeighbors == 3){
		newGrid.at(idx) = 'l';
		return;
	}

	newGrid.at(idx) = grid.at(idx);
}

struct MyInit {
	public:
		char operator()(int i, int j, int dimX, int dimY){
			InitFunction func;
			int idx = util::getIdx(i, j, dimX);
			if(idx % func(i, j) == 0){
				return 'l';
			} else {
				return 'd';
			}
		}
};

/**
 * Some playground stuff.
 * Maybe usefull for some measurements?
 */
template<typename DType, typename Stencil>
class GameOfLife {

	public:
		GameOfLife(int numX, int numY) : dimX(numX), dimY(numY), gridA(dimX*dimY), gridB(dimX*dimY), s(dimX, dimY){}

		template<typename CallableInitFunc>
		void init(CallableInitFunc f);

		void print(std::ostream &out);

		void tick();


		int dimX, dimY;
		std::vector<DType> gridA, gridB;
		Stencil s;
};

template<typename DType, typename Stencil>
void GameOfLife<DType, Stencil>::tick(){
	for(int i = 0; i < dimX; ++i){
		for(int j = 0; j < dimY; ++j){
			s.apply(i, j, gridA, gridB);
		}
	}

	std::swap(gridA, gridB);
}

template<typename DType, typename Stencil>
template<typename CallableInitFunc>
void GameOfLife<DType, Stencil>::init(CallableInitFunc f){
	for(int i = 0; i < dimX; ++i){
		for(int j = 0; j < dimY; ++j){
			int idx = util::getIdx(i, j, dimX);
			gridA.at(idx) = f(i, j, dimX, dimY);
		}
	}
}

template<typename DType, typename Stencil>
void GameOfLife<DType, Stencil>::print(std::ostream &out){
	for(int i = 0; i < dimX; ++i){
		for(int j = 0; j < dimY; ++j){
			out << gridA.at(util::getIdx(i, j, dimX));
		}
		out << "\n";
	}
	out << std::endl;
}


