#
#  Makefile for pyDIA
#

targets	: CPU GPU

CPU	: DIA_CPU.py c_functions.so

GPU	: DIA_GPU.py

DIA_CPU.py	: DIA_common.py DIA_CPU_header.py
	cat DIA_CPU_header.py DIA_common.py > DIA_CPU.py

c_functions.so	: c_functions.c
	gcc -fPIC -shared -o c_functions.so c_functions.c

DIA_GPU.py	: DIA_common.py DIA_GPU_header.py
	cat DIA_GPU_header.py DIA_common.py > DIA_GPU.py

clean	:
	rm DIA_CPU.py DIA_GPU.py c_functions.so *.pyc
