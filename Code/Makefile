#
#  Makefile for pyDIA
#

targets	: CPU_single CPU_double GPU_single GPU_double

CPU_single	: DIA_CPU.py c_functions_sp.so

CPU_double	: DIA_CPU.py c_functions_dp.so

GPU_double	: DIA_GPU.py 
	rm -f cuda_functions.py c_functions_dp.so
	ln -s cuda_functions_dp.py cuda_functions.py

GPU_single	: DIA_GPU.py c_functions_sp.so
	rm -f cuda_functions.py
	ln -s cuda_functions_sp.py cuda_functions.py

DIA_CPU.py	: DIA_common.py DIA_CPU_header.py
	cat DIA_CPU_header.py DIA_common.py > DIA_CPU.py

DIA_GPU.py	: DIA_common.py DIA_GPU_header.py
	cat DIA_GPU_header.py DIA_common.py > DIA_GPU.py

c_functions_sp.so	: c_functions_sp.c
	gcc -fPIC -shared -o c_functions_sp.so c_functions_sp.c
	rm -f c_functions.so
	ln -s c_functions_sp.so c_functions.so

c_functions_dp.so	: c_functions_dp.c
	gcc -fPIC -shared -o c_functions_dp.so c_functions_dp.c
	rm -f c_functions.so
	ln -s c_functions_dp.so c_functions.so

clean	:
	rm DIA_CPU.py DIA_GPU.py cuda_functions.py c_functions.so *.pyc
