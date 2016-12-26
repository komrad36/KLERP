EXECUTABLE_NAME=KLERP
CPP=g++
INC=
CPPFLAGS=-Wall -Wextra -Werror -Wshadow -pedantic -Ofast -std=gnu++14 -fomit-frame-pointer -mavx2 -march=native -mfma -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize
LIBS=-lpthread
CPPSOURCES=main.cpp

.PHONY : all
all: $(CPPSOURCES) $(EXECUTABLE_NAME)

$(EXECUTABLE_NAME) : $(CPPSOURCES)
	$(CPP) $(CPPFLAGS) $(CPPSOURCES) -o $@ $(LIBS)

.PHONY : clean
clean:
	rm -rf $(EXECUTABLE_NAME)
