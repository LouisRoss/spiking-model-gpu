IDIR=../../../include
ICCDIR=../../include/CommandControlAcceptors
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


_DEPS = CudaApi.h GpuModelCarrier.h GpuModelHelper.h ModelEngineContext.h ModelEngine.h ModelEngineThread.h ModelRunner.h ConfigurationRepository.h NeuronCommon.h NeuronModel.h NeuronNode.h NeuronRecord.h NeuronSynapse.h SpikeSignalProtocol.h 
DEPS = $(patsubst %,$(ILDIR)/%,$(_DEPS))

_COMMANDCONTROLDEPS = ICommandControlAcceptor.h 
COMMANDCONTROLDEPS = $(patsubst %,$(ICCDIR)/%,$(_COMMANDCONTROLDEPS))

_COMMANDCONTROLCONSOLEDEPS = CommandControlConsoleUi.h GpuModelUi.h 
COMMANDCONTROLCONSOLEDEPS = $(patsubst %,$(ICCDIR)/%,$(_COMMANDCONTROLCONSOLEDEPS))

_COMMANDCONTROLBASICDEPS = CommandControlBasicUi.h
COMMANDCONTROLBASICDEPS = $(patsubst %,$(ICCDIR)/%,$(_COMMANDCONTROLBASICDEPS))

_COMMANDCONTROLSOCKETDEPS = QueryResponseListenSocket.h QueryResponseSocket.h
COMMANDCONTROLSOCKETDEPS = $(patsubst %,$(ICCDIR)/%,$(_COMMANDCONTROLSOCKETDEPS))

_LIBSOCKETOBJS = libinetsocket.o dgramclient.o dgramoverstream.o exception.o framing.o inetbase.o inetclientdgram.o inetclientstream.o inetdgram.o inetserverdgram.o inetserverstream.o select.o socket.o streamclient.o
LIBSOCKETOBJS = $(patsubst %,$(LIBSOCKETODIR)/%,$(_LIBSOCKETOBJS))


all: $(BDIR)/CommandControlBasicUi.so $(BDIR)/CommandControlConsoleUi.so $(BDIR)/CommandControlSocket.so 
.PHONY: all

$(BDIR)/CommandControlBasicUi.so: $(ODIR)/CommandControlBasicUi.o
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(BDIR)/CommandControlConsoleUi.so: $(ODIR)/CommandControlConsoleUi.o
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(LIBSOCKETODIR)/%.o: $(LIBSOCKETCCDIR)/%.cpp
	make --directory ../libsocket/

$(LIBSOCKETODIR)/libinetsocket.o: $(LIBSOCKETCDIR)/inet/libinetsocket.c
	make --directory ../libsocket/

$(BDIR)/CommandControlSocket.so: $(ODIR)/CommandControlSocket.o $(LIBSOCKETOBJS)
	$(CC) -shared -o $@ $^ $(DEBUGFLAG) $(LFLAGS) $(LIBS)

$(ODIR)/CommandControlConsoleUi.o: CommandControlConsoleUi.cpp $(DEPS) $(COMMANDCONTROLDEPS) $(COMMANDCONTROLCONSOLEDEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(DEBUGFLAG) $(CCFLAGS) $(LOG)

$(ODIR)/CommandControlBasicUi.o: CommandControlBasicUi.cpp $(DEPS) $(COMMANDCONTROLDEPS) $(COMMANDCONTROLBASICDEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(DEBUGFLAG) $(CCFLAGS) $(LOG)

$(ODIR)/CommandControlSocket.o: CommandControlSocket.cpp $(DEPS) $(COMMANDCONTROLDEPS) $(COMMANDCONTROLSOCKETDEPS)
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
