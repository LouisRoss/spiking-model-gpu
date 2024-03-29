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

_COREUIS = Makefile mec.py mem.py mes.py mev.py nnclean.py nngenanticipate.py nngenlayer.py nnplot.py nnpost.py nn.py nntidy.py np.py ns.py
COREUIS = $(patsubst %,$(COREDIR)/ui/%,$(_COREUIS))

_MODELINCS = ConfigurationRepository.h KeyListener.h Log.h CoreCommon.h Converters.h ModelMapper.h ModelContext.h ModelInitializerProxy.h IQueryHandler.h CommandControlHandler.h NeuronRecordCommon.h Recorder.h SensorInputProxy.h SpikeOutputProxy.h Performance.h WorkerThread.h SpikeSignalProtocol.h IModelHelper.h 
MODELINCS = $(patsubst %,$(COREDIR)/include/%,$(_MODELINCS))

_MODELCOMMANDCONTROLS = CommandControlBasicUi.h CommandControlConsoleUi.h GpuModelUi.h ICommandControlAcceptor.h QueryResponseListenSocket.h QueryResponseSocket.h
MODELCOMMANDCONTROLS = $(patsubst %,$(COREDIR)/include/CommandControlAcceptors/%,$(_MODELCOMMANDCONTROLS))

_MODELINITS = IModelInitializer.h ModelAnticipateInitializer.h ModelInitializer.h ModelLayerInitializer.h ModelLifeInitializer.h ModelNeuronInitializer.h ModelSonataInitializer.h ParticleModelInitializer.h ModelPackageInitializer.h PackageInitializerDataSocket.h PackageInitializerProtocol.h
MODELINITS = $(patsubst %,$(COREDIR)/include/Initializers/%,$(_MODELINITS))

_MODELSENSORS = ISensorInput.h SensorInputFile.h SensorInputSocket.h SensorInputListenSocket.h SensorInputDataSocket.h SensorSonataFile.h 
MODELSENSORS = $(patsubst %,$(COREDIR)/include/SensorInputs/%,$(_MODELSENSORS))

_MODELOUTPUTS = ISpikeOutput.h SpikeOutputRecord.h SpikeOutputSocket.h 
MODELOUTPUTS = $(patsubst %,$(COREDIR)/include/SpikeOutputs/%,$(_MODELOUTPUTS))


all: $(CORECONFIGS) $(COREUIS) $(MODELINCS) $(MODELINITS) $(MODELSENSORS) $(MODELOUTPUTS) $(MODELCOMMANDCONTROLS)
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
	cp spiking-model-gpu/ui/Makefile $(COREDIR)/ui/

$(COREDIR)/ui/mec.py: spiking-model-gpu//ui/mec.py
	cp spiking-model-gpu/ui/mec.py $(COREDIR)/ui/

$(COREDIR)/ui/mem.py: spiking-model-gpu//ui/mem.py
	cp spiking-model-gpu/ui/mem.py $(COREDIR)/ui/

$(COREDIR)/ui/mes.py: spiking-model-gpu//ui/mes.py
	cp spiking-model-gpu/ui/mes.py $(COREDIR)/ui/

$(COREDIR)/ui/mev.py: spiking-model-gpu//ui/mev.py
	cp spiking-model-gpu/ui/mev.py $(COREDIR)/ui/

$(COREDIR)/ui/nnclean.py: spiking-model-gpu//ui/nnclean.py
	cp spiking-model-gpu/ui/nnclean.py $(COREDIR)/ui/

$(COREDIR)/ui/nngenanticipate.py: spiking-model-gpu//ui/nngenanticipate.py
	cp spiking-model-gpu/ui/nngenanticipate.py $(COREDIR)/ui/

$(COREDIR)/ui/nngenlayer.py: spiking-model-gpu//ui/nngenlayer.py
	cp spiking-model-gpu/ui/nngenlayer.py $(COREDIR)/ui/

$(COREDIR)/ui/nnplot.py: spiking-model-gpu//ui/nnplot.py
	cp spiking-model-gpu/ui/nnplot.py $(COREDIR)/ui/

$(COREDIR)/ui/nnpost.py: spiking-model-gpu//ui/nnpost.py
	cp spiking-model-gpu/ui/nnpost.py $(COREDIR)/ui/

$(COREDIR)/ui/nn.py: spiking-model-gpu//ui/nn.py
	cp spiking-model-gpu/ui/nn.py $(COREDIR)/ui/

$(COREDIR)/ui/nntidy.py: spiking-model-gpu//ui/nntidy.py
	cp spiking-model-gpu/ui/nntidy.py $(COREDIR)/ui/

$(COREDIR)/ui/np.py: spiking-model-gpu//ui/np.py
	cp spiking-model-gpu/ui/np.py $(COREDIR)/ui/

$(COREDIR)/ui/ns.py: spiking-model-gpu//ui/ns.py
	cp spiking-model-gpu/ui/ns.py $(COREDIR)/ui/


### Reverse the path of files obtained through modelengine/add-dependencies

# Some individual files from the include folder
$(COREDIR)/include/ConfigurationRepository.h: spiking-model-gpu/include/ConfigurationRepository.h
	cp spiking-model-gpu/include/ConfigurationRepository.h $(COREDIR)/include/

$(COREDIR)/include/KeyListener.h: spiking-model-gpu/include/KeyListener.h
	cp spiking-model-gpu/include/KeyListener.h $(COREDIR)/include/

$(COREDIR)/include/Log.h: spiking-model-gpu/include/Log.h
	cp spiking-model-gpu/include/Log.h $(COREDIR)/include/

$(COREDIR)/include/CoreCommon.h: spiking-model-gpu/include/CoreCommon.h
	cp spiking-model-gpu/include/CoreCommon.h $(COREDIR)/include/

$(COREDIR)/include/Converters.h: spiking-model-gpu/include/Converters.h
	cp spiking-model-gpu/include/Converters.h $(COREDIR)/include/

$(COREDIR)/include/ModelMapper.h: spiking-model-gpu/include/ModelMapper.h
	cp spiking-model-gpu/include/ModelMapper.h $(COREDIR)/include/

$(COREDIR)/include/ModelContext.h: spiking-model-gpu/include/ModelContext.h
	cp spiking-model-gpu/include/ModelContext.h $(COREDIR)/include/

$(COREDIR)/include/ModelInitializerProxy.h: spiking-model-gpu/include/ModelInitializerProxy.h
	cp spiking-model-gpu/include/ModelInitializerProxy.h $(COREDIR)/include/

$(COREDIR)/include/IQueryHandler.h: spiking-model-gpu/include/IQueryHandler.h
	cp spiking-model-gpu/include/IQueryHandler.h $(COREDIR)/include/

$(COREDIR)/include/CommandControlHandler.h: spiking-model-gpu/include/CommandControlHandler.h
	cp spiking-model-gpu/include/CommandControlHandler.h $(COREDIR)/include/

$(COREDIR)/include/Performance.h: spiking-model-gpu/include/Performance.h
	cp spiking-model-gpu/include/Performance.h $(COREDIR)/include/

$(COREDIR)/include/WorkerThread.h: spiking-model-gpu/include/WorkerThread.h
	cp spiking-model-gpu/include/WorkerThread.h $(COREDIR)/include/

$(COREDIR)/include/IModelRunner.h: spiking-model-gpu/include/IModelRunner.h
	cp spiking-model-gpu/include/IModelRunner.h $(COREDIR)/include/

$(COREDIR)/include/NeuronRecordCommon.h: spiking-model-gpu/include/NeuronRecordCommon.h
	cp spiking-model-gpu/include/NeuronRecordCommon.h $(COREDIR)/include/

$(COREDIR)/include/Recorder.h: spiking-model-gpu/include/Recorder.h
	cp spiking-model-gpu/include/Recorder.h $(COREDIR)/include/

$(COREDIR)/include/SensorInputProxy.h: spiking-model-gpu/include/SensorInputProxy.h
	cp spiking-model-gpu/include/SensorInputProxy.h $(COREDIR)/include/

$(COREDIR)/include/SpikeOutputProxy.h: spiking-model-gpu/include/SpikeOutputProxy.h
	cp spiking-model-gpu/include/SpikeOutputProxy.h $(COREDIR)/include/

$(COREDIR)/include/SpikeSignalProtocol.h: spiking-model-gpu/include/SpikeSignalProtocol.h
	cp spiking-model-gpu/include/SpikeSignalProtocol.h $(COREDIR)/include/

$(COREDIR)/include/IModelHelper.h: spiking-model-gpu/include/IModelHelper.h
	cp spiking-model-gpu/include/IModelHelper.h $(COREDIR)/include/

# All files from the include/CommandControlAcceptors folder
$(COREDIR)/include/CommandControlAcceptors/ICommandControlAcceptor.h: spiking-model-gpu/include/CommandControlAcceptors/ICommandControlAcceptor.h
	cp spiking-model-gpu/include/CommandControlAcceptors/ICommandControlAcceptor.h $(COREDIR)/include/CommandControlAcceptors/

$(COREDIR)/include/CommandControlAcceptors/CommandControlConsoleUi.h: spiking-model-gpu/include/CommandControlAcceptors/CommandControlConsoleUi.h
	cp spiking-model-gpu/include/CommandControlAcceptors/CommandControlConsoleUi.h $(COREDIR)/include/CommandControlAcceptors/

$(COREDIR)/include/CommandControlAcceptors/GpuModelUi.h: spiking-model-gpu/include/CommandControlAcceptors/GpuModelUi.h
	cp spiking-model-gpu/include/CommandControlAcceptors/GpuModelUi.h $(COREDIR)/include/CommandControlAcceptors/

$(COREDIR)/include/CommandControlAcceptors/QueryResponseListenSocket.h: spiking-model-gpu/include/CommandControlAcceptors/QueryResponseListenSocket.h
	cp spiking-model-gpu/include/CommandControlAcceptors/QueryResponseListenSocket.h $(COREDIR)/include/CommandControlAcceptors/

$(COREDIR)/include/CommandControlAcceptors/QueryResponseSocket.h: spiking-model-gpu/include/CommandControlAcceptors/QueryResponseSocket.h
	cp spiking-model-gpu/include/CommandControlAcceptors/QueryResponseSocket.h $(COREDIR)/include/CommandControlAcceptors/

$(COREDIR)/include/CommandControlAcceptors/CommandControlBasicUi.h: spiking-model-gpu/include/CommandControlAcceptors/CommandControlBasicUi.h
	cp spiking-model-gpu/include/CommandControlAcceptors/CommandControlBasicUi.h $(COREDIR)/include/CommandControlAcceptors/

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

$(COREDIR)/include/Initializers/ModelPackageInitializer.h: spiking-model-gpu/include/Initializers/ModelPackageInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelPackageInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ModelSonataInitializer.h: spiking-model-gpu/include/Initializers/ModelSonataInitializer.h
	cp spiking-model-gpu/include/Initializers/ModelSonataInitializer.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/PackageInitializerDataSocket.h: spiking-model-gpu/include/Initializers/PackageInitializerDataSocket.h
	cp spiking-model-gpu/include/Initializers/PackageInitializerDataSocket.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/PackageInitializerProtocol.h: spiking-model-gpu/include/Initializers/PackageInitializerProtocol.h
	cp spiking-model-gpu/include/Initializers/PackageInitializerProtocol.h $(COREDIR)/include/Initializers/

$(COREDIR)/include/Initializers/ParticleModelInitializer.h: spiking-model-gpu/include/Initializers/ParticleModelInitializer.h
	cp spiking-model-gpu/include/Initializers/ParticleModelInitializer.h $(COREDIR)/include/Initializers/


# All files from the include/SensorInputs folder
$(COREDIR)/include/SensorInputs/ISensorInput.h: spiking-model-gpu/include/SensorInputs/ISensorInput.h
	cp spiking-model-gpu/include/SensorInputs/ISensorInput.h $(COREDIR)/include/SensorInputs/

$(COREDIR)/include/SensorInputs/SensorInputDataSocket.h: spiking-model-gpu/include/SensorInputs/SensorInputDataSocket.h
	cp spiking-model-gpu/include/SensorInputs/SensorInputDataSocket.h $(COREDIR)/include/SensorInputs/

$(COREDIR)/include/SensorInputs/SensorInputFile.h: spiking-model-gpu/include/SensorInputs/SensorInputFile.h
	cp spiking-model-gpu/include/SensorInputs/SensorInputFile.h $(COREDIR)/include/SensorInputs/

$(COREDIR)/include/SensorInputs/SensorInputListenSocket.h: spiking-model-gpu/include/SensorInputs/SensorInputListenSocket.h
	cp spiking-model-gpu/include/SensorInputs/SensorInputListenSocket.h $(COREDIR)/include/SensorInputs/

$(COREDIR)/include/SensorInputs/SensorInputSocket.h: spiking-model-gpu/include/SensorInputs/SensorInputSocket.h
	cp spiking-model-gpu/include/SensorInputs/SensorInputSocket.h $(COREDIR)/include/SensorInputs/

$(COREDIR)/include/SensorInputs/SensorSonataFile.h: spiking-model-gpu/include/SensorInputs/SensorSonataFile.h
	cp spiking-model-gpu/include/SensorInputs/SensorSonataFile.h $(COREDIR)/include/SensorInputs/

# All files from the include/SpikeOutputs folder
$(COREDIR)/include/SpikeOutputs/ISpikeOutput.h: spiking-model-gpu/include/SpikeOutputs/ISpikeOutput.h
	cp spiking-model-gpu/include/SpikeOutputs/ISpikeOutput.h $(COREDIR)/include/SpikeOutputs/

$(COREDIR)/include/SpikeOutputs/SpikeOutputRecord.h: spiking-model-gpu/include/SpikeOutputs/SpikeOutputRecord.h
	cp spiking-model-gpu/include/SpikeOutputs/SpikeOutputRecord.h $(COREDIR)/include/SpikeOutputs/

$(COREDIR)/include/SpikeOutputs/SpikeOutputSocket.h: spiking-model-gpu/include/SpikeOutputs/SpikeOutputSocket.h
	cp spiking-model-gpu/include/SpikeOutputs/SpikeOutputSocket.h $(COREDIR)/include/SpikeOutputs/

