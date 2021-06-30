IDIR=../../../include
ILDIR=../../include
CC=g++
CCFLAGS=-I$(IDIR) -I$(ILDIR) -fpic -DMIXED -pthread -std=c++17
LFLAGS=-fpic -pthread -std=c++17
LOG=-DNOLOG
DEBUGFLAG=

ODIR=obj
LIBSOCKETODIR=../libsocket/obj
LIBSOCKETCCDIR=../libsocket/C++
LIBSOCKETCDIR=../libsocket/C
BDIR=../../bin
LDIR=../../lib

LIBS= -ldl -ltbb


_DEPS = CudaApi.h GpuModelCarrier.h GpuModelHelper.h GpuModelUi.h ModelEngineContext.h ModelEngine.h ModelEngineThread.h ModelRunner.h NeuronCommon.h NeuronModel.h NeuronNode.h NeuronRecord.h NeuronSynapse.h
DEPS = $(patsubst %,$(ILDIR)/%,$(_DEPS))

_SPIKEOUTPUTDEPS = ISpikeOutput.h  SpikeOutputRecord.h SpikeOutputSocket.h 
SPIKEOUTPUTDEPS = $(patsubst %,$(ILDIR)/SpikeOutputs/%,$(_SPIKEOUTPUTDEPS))

_LIBSOCKETOBJS = libinetsocket.o dgramclient.o dgramoverstream.o exception.o framing.o inetbase.o inetclientdgram.o inetclientstream.o inetdgram.o inetserverdgram.o inetserverstream.o select.o socket.o streamclient.o
LIBSOCKETOBJS = $(patsubst %,$(LIBSOCKETODIR)/%,$(_LIBSOCKETOBJS))


all: $(BDIR)/SpikeOutputRecord.so $(BDIR)/SpikeOutputSocket.so 
.PHONY: all

$(BDIR)/SpikeOutputRecord.so: $(ODIR)/SpikeOutputRecord.o
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(LIBSOCKETODIR)/%.o: $(LIBSOCKETCCDIR)/%.cpp
	make --directory ../libsocket/

$(LIBSOCKETODIR)/libinetsocket.o: $(LIBSOCKETCDIR)/inet/libinetsocket.c
	make --directory ../libsocket/

$(BDIR)/SpikeOutputSocket.so: $(ODIR)/SpikeOutputSocket.o $(LIBSOCKETOBJS)
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(ODIR)/%.o: %.cpp $(DEPS) $(SPIKEOUTPUTDEPS)
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
