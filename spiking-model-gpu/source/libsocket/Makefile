LIBSOCKETINCLUDE=-I../../../include/libsocket
ODIR=obj

C=gcc
CC=g++
IDIR=../../include
ILDIR=../../../include
CCDIR=./C++
CDIR=./C

ALL_CCFLAGS :=
ALL_CCFLAGS += -std=c++17
ALL_CCFLAGS += -fpic 

ALL_CFLAGS :=
ALL_CFLAGS += -fpic

DEBUGFLAG=

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)


_LIBSOCKETOBJS = libinetsocket.o dgramclient.o dgramoverstream.o exception.o framing.o inetbase.o inetclientdgram.o inetclientstream.o inetdgram.o inetserverdgram.o inetserverstream.o select.o socket.o streamclient.o
LIBSOCKETOBJS = $(patsubst %,$(ODIR)/%,$(_LIBSOCKETOBJS))

all: $(LIBSOCKETOBJS) 

$(ODIR)/%.o: $(CCDIR)/%.cpp
	mkdir -p $(ODIR)
	$(CC) $(DEBUGFLAG) $(LIBSOCKETINCLUDE) $(ALL_CCFLAGS) -DMIXED -c -o $@ $<

$(ODIR)/libinetsocket.o: $(CDIR)/inet/libinetsocket.c
	mkdir -p $(ODIR)
	$(C) $(DEBUGFLAG) $(LIBSOCKETINCLUDE) $(ALL_CFLAGS) -DMIXED -c -o $@ $<

debug:
	make DEBUGFLAG="-g" LOG=""
.PHONY: debug

clean:
	rm -f $(ODIR)/*o

clobber: clean
