DEBUGFLAG=
BDIR=../bin

all:
	cp -u ModelSettings.json $(BDIR)
	make --directory=NeuronModel DEBUGFLAG="$(DEBUGFLAG)"
	make --directory=Initializers DEBUGFLAG="$(DEBUGFLAG)"
	make --directory=SensorInputs DEBUGFLAG="$(DEBUGFLAG)"
	make --directory=SpikeOutputs DEBUGFLAG="$(DEBUGFLAG)"
.PHONY: all

debug:
	make DEBUGFLAG="-g"
.PHONY: debug

