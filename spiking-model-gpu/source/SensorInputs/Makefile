IDIR=../../../include
ILDIR=../../include
IH5DIR=/usr/include/hdf5/serial/
CC=g++
CCFLAGS=-I$(IDIR) -I$(ILDIR) -I$(IH5DIR) -fpic -DMIXED -pthread -std=c++17
LFLAGS=-fpic -pthread -std=c++17
LOG=-DNOLOG
DEBUGFLAG=

ODIR=obj
LIBSOCKETODIR=../libsocket/obj
LIBSOCKETCCDIR=../libsocket/C++
LIBSOCKETCDIR=../libsocket/C
BDIR=../../bin
LDIR=../../lib

LIBS= -ldl -lhdf5_serial -ltbb


_DEPS = CudaApi.h GpuModelCarrier.h GpuModelHelper.h GpuModelUi.h ModelEngineContext.h ModelEngine.h ModelEngineThread.h ModelRunner.h NeuronCommon.h NeuronModel.h NeuronNode.h NeuronRecord.h NeuronSynapse.h SpikeSignalProtocol.h 
DEPS = $(patsubst %,$(ILDIR)/%,$(_DEPS))

_SENSORINPUTDEPS = ISensorInput.h  SensorInputFile.h SensorInputSocket.h SensorInputListenSocket.h SensorInputDataSocket.h  SensorSonataFile.h 
SENSORINPUTDEPS = $(patsubst %,$(ILDIR)/SensorInputs/%,$(_SENSORINPUTDEPS))

_SONATADEPS = SonataCompiledModel.h SonataEdgeModel.h SonataEdgeModelLoader.h SonataInputSpikeLoader.h SonataModelLoader.h SonataModelPersister.h SonataModelRepository.h SonataNodeModel.h SonataNodeModelLoader.h
SONATADEPS = $(patsubst %,$(ILDIR)/persistence/sonata/%,$(_SONATADEPS))

_LIBSOCKETOBJS = libinetsocket.o dgramclient.o dgramoverstream.o exception.o framing.o inetbase.o inetclientdgram.o inetclientstream.o inetdgram.o inetserverdgram.o inetserverstream.o select.o socket.o streamclient.o
LIBSOCKETOBJS = $(patsubst %,$(LIBSOCKETODIR)/%,$(_LIBSOCKETOBJS))


all: $(BDIR)/SensorInputFile.so $(BDIR)/SensorInputSocket.so 
.PHONY: all

$(BDIR)/SensorInputFile.so: $(ODIR)/SensorInputFile.o
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(LIBSOCKETODIR)/%.o: $(LIBSOCKETCCDIR)/%.cpp
	make --directory ../libsocket/

$(LIBSOCKETODIR)/libinetsocket.o: $(LIBSOCKETCDIR)/inet/libinetsocket.c
	make --directory ../libsocket/

$(BDIR)/SensorInputSocket.so: $(ODIR)/SensorInputSocket.o $(LIBSOCKETOBJS)
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
