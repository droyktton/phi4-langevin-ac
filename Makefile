# Builds the Langevin + AC-field solver (main.cu + langevin.h).
#
# The coefficients c, eps0 and r0 are compile-time constants (c=eps0=r0=1
# without loss of generality); everything else (field, temperature,
# disorder, time step...) is a runtime option, see ./phi4langevin --help.
# Override on the command line, e.g.:
#   make CEL=0.5 ARCH=sm_86
CEL=1.0
EPSILON0=1.0
R0=1.0
ARCH=sm_61

FLAGS=-DCEL=$(CEL) -DEPSILON0=$(EPSILON0) -DR0=$(R0)

NVCC=-std=c++14 -arch=$(ARCH)

phi4langevin: main.cu langevin.h
	nvcc $(NVCC) -O2 -I. -o phi4langevin main.cu $(FLAGS)

clean:
	rm -f phi4langevin
