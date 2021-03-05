IDIR=../../../include
ILDIR=../../include
IH5DIR=/usr/include/hdf5/serial/
CC=g++
CCFLAGS=-I$(IDIR) -I$(ILDIR) -I$(IH5DIR) -fpic -pthread -std=c++17
LFLAGS=-fpic -pthread -std=c++17
LOG=-DNOLOG
DEBUGFLAG=

ODIR=obj
BDIR=../../bin
LDIR=../../lib

LIBS= -ldl -lhdf5_serial -ltbb


_DEPS = CudaApi.h GpuModelCarrier.h GpuModelHelper.h GpuModelUi.h ModelEngineContext.h ModelEngine.h ModelEngineThread.h ModelRunner.h NeuronCommon.h NeuronModel.h NeuronNode.h NeuronRecord.h NeuronSynapse.h
DEPS = $(patsubst %,$(ILDIR)/%,$(_DEPS))

_SENSORINPUTDEPS = ISensorInput.h  SensorInputFile.h  SensorSonataFile.h
SENSORINPUTDEPS = $(patsubst %,$(ILDIR)/SensorInputs/%,$(_SENSORINPUTDEPS))

_SONATADEPS = SonataCompiledModel.h SonataEdgeModel.h SonataEdgeModelLoader.h SonataInputSpikeLoader.h SonataModelLoader.h SonataModelPersister.h SonataModelRepository.h SonataNodeModel.h SonataNodeModelLoader.h
SONATADEPS = $(patsubst %,$(ILDIR)/persistence/sonata/%,$(_SONATADEPS))

all: $(BDIR)/SensorInputFile.so 
.PHONY: all

$(BDIR)/SensorInputFile.so: $(ODIR)/SensorInputFile.o
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(BDIR)/SensorSonataFile.so: $(ODIR)/SensorSonataFile.o
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(ODIR)/%.o: %.cpp $(DEPS) $(SENSORINPUTDEPS) $(SONATADEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(DEBUGFLAG) $(CCFLAGS) $(LOG)

debug:
	make DEBUGFLAG="-g"
.PHONY: debug

log:
	make LOG=""
.PHONY: log

clean:
	rm -f $(ODIR)/*.o
.PHONY: clean
