# This is a makefile intended to copy any files originally obtained
# from another repo back to those repos, so their changed status
# can be ascertained, and proper commits can be done there.
# Basically, files obtained this way are obtained using the 
# add-dependencies script, so this is intended to reverse those copies.
# As such, it is expected that the repositories we depend on are present
# in parallel folders to this one, and up to date before running this.
#
# use:
# >make -f backfill

COREDIR=../spiking-model-core

_CORECONFIGS = a1.json bmtk1.json l1.json Anticipate/Anticipate.json Anticipate/I1-I2-N1-N2.json BMTK1/bmtk.json BMTK1/2-5.json Layer/Layer.json Layer/Layer1-5.json
CORECONFIGS = $(patsubst %,$(COREDIR)/config/%,$(_CORECONFIGS))

_COREUIS = Makefile mec.py mem.py mes.py mev.py nnclean.py nngenanticipate.py nngenlayer.py nnplot.py nn.py nntidy.py np.py 
COREUIS = $(patsubst %,$(COREDIR)/ui/%,$(_COREUIS))

_MODELINCS = ConfigurationRepository.h KeyListener.h Log.h ModelInitializerProxy.h CommandControlConsoleUi.h ICommandControlAcceptor.h QueryResponseSocket.h QueryResponseListenSocket.h IQueryHandler.h CommandControlHandler.h NeuronRecordCommon.h Recorder.h SensorInputProxy.h Performance.h 
MODELINCS = $(patsubst %,$(COREDIR)/include/%,$(_MODELINCS))

_MODELINITS = IModelInitializer.h ModelAnticipateInitializer.h ModelInitializer.h ModelLayerInitializer.h ModelLifeInitializer.h ModelNeuronInitializer.h ModelSonataInitializer.h ParticleModelInitializer.h 
MODELINITS = $(patsubst %,$(COREDIR)/include/Initializers/%,$(_MODELINITS))

_MODELSENSORS = ISensorInput.h SensorInputFile.h SensorSonataFile.h 
MODELSENSORS = $(patsubst %,$(COREDIR)/include/SensorInputs/%,$(_MODELSENSORS))


all: $(CORECONFIGS) $(COREUIS) $(MODELINCS) $(MODELINITS) $(MODELSENSORS) 
.PHONY: all

# All files in the config folder
$(COREDIR)/config/a1.json: spiking-model-gpu//config/a1.json
	cp spiking-model-gpu//config/a1.json $(COREDIR)/config/

$(COREDIR)/config/bmtk1.json: spiking-model-gpu//config/bmtk1.json
	cp spiking-model-gpu//config/bmtk1.json $(COREDIR)/config/

$(COREDIR)/config/l1.json: spiking-model-gpu//config/l1.json
	cp spiking-model-gpu//config/l1.json $(COREDIR)/config/

$(COREDIR)/config/Anticipate/Anticipate.json: spiking-model-gpu//config/Anticipate/Anticipate.json
	cp spiking-model-gpu//config/Anticipate/Anticipate.json $(COREDIR)/config/Anticipate/

$(COREDIR)/config/Anticipate/I1-I2-N1-N2.json: spiking-model-gpu//config/Anticipate/I1-I2-N1-N2.json
	cp spiking-model-gpu//config/Anticipate/I1-I2-N1-N2.json $(COREDIR)/config/Anticipate/

$(COREDIR)/config/BMTK1/bmtk.json: spiking-model-gpu//config/BMTK1/bmtk.json
	cp spiking-model-gpu//config/BMTK1/bmtk.json $(COREDIR)/config/BMTK1/

$(COREDIR)/config/BMTK1/2-5.json: spiking-model-gpu//config/BMTK1/2-5.json
	cp spiking-model-gpu//config/BMTK1/2-5.json $(COREDIR)/config/BMTK1/

$(COREDIR)/config/Layer/Layer.json: spiking-model-gpu//config/Layer/Layer.json
	cp spiking-model-gpu//config/Layer/Layer.json $(COREDIR)/config/Layer/

$(COREDIR)/config/Layer/Layer1-5.json: spiking-model-gpu//config/Layer/Layer1-5.json
	cp spiking-model-gpu//config/Layer/Layer1-5.json $(COREDIR)/config/Layer/

# All files in the ui folder
$(COREDIR)/ui/Makefile: spiking-model-gpu//ui/Makefile
	cp spiking-model-gpu//ui/Makefile $(COREDIR)/ui/

$(COREDIR)/ui/mec.py: spiking-model-gpu//ui/mec.py
	cp spiking-model-gpu//ui/mec.py $(COREDIR)/ui/

$(COREDIR)/ui/mem.py: spiking-model-gpu//ui/mem.py
	cp spiking-model-gpu//ui/mem.py $(COREDIR)/ui/

$(COREDIR)/ui/mes.py: spiking-model-gpu//ui/mes.py
	cp spiking-model-gpu//ui/mes.py $(COREDIR)/ui/

$(COREDIR)/ui/mev.py: spiking-model-gpu//ui/mev.py
	cp spiking-model-gpu//ui/mev.py $(COREDIR)/ui/

$(COREDIR)/ui/nnclean.py: spiking-model-gpu//ui/nnclean.py
	cp spiking-model-gpu//ui/nnclean.py $(COREDIR)/ui/

$(COREDIR)/ui/nngenanticipate.py: spiking-model-gpu//ui/nngenanticipate.py
	cp spiking-model-gpu//ui/nngenanticipate.py $(COREDIR)/ui/

$(COREDIR)/ui/nngenlayer.py: spiking-model-gpu//ui/nngenlayer.py
	cp spiking-model-gpu//ui/nngenlayer.py $(COREDIR)/ui/

$(COREDIR)/ui/nnplot.py: spiking-model-gpu//ui/nnplot.py
	cp spiking-model-gpu//ui/nnplot.py $(COREDIR)/ui/

$(COREDIR)/ui/nn.py: spiking-model-gpu//ui/nn.py
	cp spiking-model-gpu//ui/nn.py $(COREDIR)/ui/

$(COREDIR)/ui/nntidy.py: spiking-model-gpu//ui/nntidy.py
	cp spiking-model-gpu//ui/nntidy.py $(COREDIR)/ui/

$(COREDIR)/ui/np.py: spiking-model-gpu//ui/np.py
	cp spiking-model-gpu//ui/np.py $(COREDIR)/ui/


### Reverse the path of files obtained through modelengine/add-dependencies

# Some individual files from the include folder
$(COREDIR)/include/ConfigurationRepository.h: spiking-model-gpu/include/ConfigurationRepository.h
	cp spiking-model-gpu/include/ConfigurationRepository.h $(COREDIR)/include/

$(COREDIR)/include/KeyListener.h: spiking-model-gpu/include/KeyListener.h
	cp spiking-model-gpu/include/KeyListener.h $(COREDIR)/include/

$(COREDIR)/include/Log.h: spiking-model-gpu/include/Log.h
	cp spiking-model-gpu/include/Log.h $(COREDIR)/include/

$(COREDIR)/include/ModelInitializerProxy.h: spiking-model-gpu/include/ModelInitializerProxy.h
	cp spiking-model-gpu/include/ModelInitializerProxy.h $(COREDIR)/include/

$(COREDIR)/include/CommandControlConsoleUi.h: spiking-model-gpu/include/CommandControlConsoleUi.h
	cp spiking-model-gpu/include/CommandControlConsoleUi.h $(COREDIR)/include/

$(COREDIR)/include/ICommandControlAcceptor.h: spiking-model-gpu/include/ICommandControlAcceptor.h
	cp spiking-model-gpu/include/ICommandControlAcceptor.h $(COREDIR)/include/

$(COREDIR)/include/QueryResponseSocket.h: spiking-model-gpu/include/QueryResponseSocket.h
	cp spiking-model-gpu/include/QueryResponseSocket.h $(COREDIR)/include/

$(COREDIR)/include/QueryResponseListenSocket.h: spiking-model-gpu/include/QueryResponseListenSocket.h
	cp spiking-model-gpu/include/QueryResponseListenSocket.h $(COREDIR)/include/

$(COREDIR)/include/IQueryHandler.h: spiking-model-gpu/include/IQueryHandler.h
	cp spiking-model-gpu/include/IQueryHandler.h $(COREDIR)/include/

$(COREDIR)/include/CommandControlHandler.h: spiking-model-gpu/include/CommandControlHandler.h
	cp spiking-model-gpu/include/CommandControlHandler.h $(COREDIR)/include/

$(COREDIR)/include/Performance.h: spiking-model-gpu/include/Performance.h
	cp spiking-model-gpu/include/Performance.h $(COREDIR)/include/

$(COREDIR)/include/IModelRunner.h: spiking-model-gpu/include/IModelRunner.h
	cp spiking-model-gpu/include/IModelRunner.h $(COREDIR)/include/

$(COREDIR)/include/NeuronRecordCommon.h: spiking-model-gpu/include/NeuronRecordCommon.h
	cp spiking-model-gpu/include/NeuronRecordCommon.h $(COREDIR)/include/

$(COREDIR)/include/Recorder.h: spiking-model-gpu/include/Recorder.h
	cp spiking-model-gpu/include/Recorder.h $(COREDIR)/include/

$(COREDIR)/include/SensorInputProxy.h: spiking-model-gpu/include/SensorInputProxy.h
	cp spiking-model-gpu/include/SensorInputProxy.h $(COREDIR)/include/

# All files from the include/Initializers folder
$(COREDIR)/include/Initializers/IModelInitializer.h: spiking-model-gpu/include/Initializers/IModelInitializer.h
	cp spiking-model-gpu/include/Initializers/IModelInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ModelAnticipateInitializer.h: spiking-model-gpu/include/Initializers/ModelAnticipateInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelAnticipateInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ModelInitializer.h: spiking-model-gpu/include/Initializers/ModelInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ModelLayerInitializer.h: spiking-model-gpu/include/Initializers/ModelLayerInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelLayerInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ModelLifeInitializer.h: spiking-model-gpu/include/Initializers/ModelLifeInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelLifeInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ModelNeuronInitializer.h: spiking-model-gpu/include/Initializers/ModelNeuronInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelNeuronInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ModelSonataInitializer.h: spiking-model-gpu/include/Initializers/ModelSonataInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelSonataInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ParticleModelInitializer.h: spiking-model-gpu/include/Initializers/ParticleModelInitializer.h
	cp spiking-model-gpu/include/Initializers/ParticleModelInitializer.h $(COREDIR)/include/Initializers/

# All files from the include/SensorInputs folder
$(COREDIR)/include/SensorInputs/ISensorInput.h: spiking-model-gpu/include/SensorInputs/ISensorInput.h
	cp spiking-model-gpu/include/SensorInputs/ISensorInput.h $(COREDIR)/include/SensorInputs/

$(COREDIR)/include/SensorInputs/SensorInputFile.h: spiking-model-gpu/include/SensorInputs/SensorInputFile.h
	cp spiking-model-gpu/include/SensorInputs/SensorInputFile.h $(COREDIR)/include/SensorInputs/

$(COREDIR)/include/SensorInputs/SensorSonataFile.h: spiking-model-gpu/include/SensorInputs/SensorSonataFile.h
	cp spiking-model-gpu/include/SensorInputs/SensorSonataFile.h $(COREDIR)/include/SensorInputs/
