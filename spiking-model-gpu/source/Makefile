IDIR=../include
IH5DIR=/usr/include/hdf5/serial/
ICUDADIR=/usr/local/cuda/include
CC=g++
CCFLAGS=-I$(IDIR) -I$(ICUDADIR) -I$(IH5DIR) -fpic -pthread -std=c++17
LFLAGS=-fpic -pthread -std=c++17 -L$(LDIR)
LOG=-DNOLOG

ODIR=obj
BDIR=../bin
LDIR=/usr/local/cuda/lib64

LIBS= -ldl -lhdf5_serial -ltbb -lcudart


_DEPS = 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_SONATADEPS = persistence/sonata/SonataCompiledModel.h persistence/sonata/SonataEdgeModel.h persistence/sonata/SonataEdgeModelLoader.h persistence/sonata/SonataInputSpikeLoader.h persistence/sonata/SonataModelLoader.h persistence/sonata/SonataModelPersister.h persistence/sonata/SonataModelRepository.h persistence/sonata/SonataNodeModel.h persistence/sonata/SonataNodeModelLoader.h
SONATADEPS = $(patsubst %,$(IDIR)/%,$(_SONATADEPS))

_LOCALDEPS = 
LOCALDEPS = $(patsubst %,./%,$(_LOCALDEPS))

$(ODIR)/%.o: %.cpp $(DEPS) $(SONATADEPS) $(LOCALDEPS)
	$(CC) -c -o $@ $< $(CCFLAGS) $(LOG)

all: $(BDIR)/query_device
.PHONY: all

log:
	make LOG=""
.PHONY: log

$(BDIR)/query_device: $(ODIR)/query_device.o
	$(CC) -o $@ $^ $(LFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
