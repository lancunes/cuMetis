all:
	nvcc -std=c++11 -gencode arch=compute_80,code=sm_80 -O3  cuMetis.cu -o  cuMetis  --expt-relaxed-constexpr -w
	gcc mtx_to_graph.c -o mtx_to_graph
