NMC Aging Dataset README
This README provides an overview of the NMC (Nickel Manganese Cobalt) cell aging dataset, which consists of data collected from 48 NMC cells subjected to aging until reaching 60% State of Health (SOH). The dataset was gathered from the study conducted by Baumhöfer et al. (2014), referenced as Ref 1.

Data Files:
charge_voltage_profiles.pkl: This file contains the voltage response corresponding to Constant Current Constant Voltage (CCCV) charging profiles at all Reference Performance Tests (RPTs) for all cells.

pulse_voltage_profiles.pkl: Contains the voltage response corresponding to pulse current profiles at all RPTs for all cells.

pulse_resistances.pkl: Includes discharge capacity and resistance data at all RPTs for all cells.

Applied Pulses:
Pulse experiments were conducted at three different current levels and three State of Charges (SOCs):

	Current Levels:
	C3: 2 A
	C4: 4 A
	C5: 6 A
	
	SOCs:
	30soc: 30% State of Charge
	50soc: 50% State of Charge
	90soc: 90% State of Charge

State Values:
Each RPT entry includes ten state values, which are represented as follows:
	soh: State of Health
	resistances: Resistance values in the form 'r' followed by the current level and SOC
	
Reference:
Baumhöfer, T., Brühl, M., Rothgang, S., & Sauer, D. U. (2014). Production caused variation in capacity aging trend and correlation to initial cell performance. Journal of Power Sources, 247, 332–338. https://doi.org/10.1016/j.jpowsour.2013.08.108

This dataset provides valuable insights into the aging behavior of NMC cells and its correlation with initial cell performance.
