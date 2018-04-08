MDRFSimulated_252X252_NumPhoton200000_NumSiPMs20_CsI_Barrier0_QE0.2_lambertianRef0.9_gelIndex1.830_WholeLargeRodSpace1_.bin
	The above file is the MDRF which characterize the detector's mean response to gamma-ray interaction position. Without optical barriers
7lineSimulated.bin: This file is the simulated events for evaluating the performance of neural networks. Without optical barriers

MDRFSimulated_252X252_NumPhoton200000_NumSiPMs20_CsI_Barrier1_QE0.2_lambertianRef0.9_gelIndex1.500_WholeLargeRodSpace1_.bin
	The above file is the MDRF which characterize the detector's mean response to gamma-ray interaction position. With optical barriers
5lineSimulated.bin: This file is the simulated events for evaluating the performance of neural networks. With optical barriers

trainXYPatchNetG.py is responsible to train the global estimation network which can give a rough estimate of each event's position
trainXYPatchNetL.py is responsible to launch a neural network array, to estimate each event's position in a local scale
estimateXYPatchNet.py is responsible for estimation of each event's position from the simulated data

The folders: model_GX2 is the second version of the neural network trained for rough estimation in X direction
model_LX3 is the third version of the neural network arrays trained for X direction