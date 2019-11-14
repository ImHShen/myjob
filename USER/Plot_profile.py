from hapi import *
import matplotlib.pyplot as plt
db_begin('data')
#fetch('HF',14,1,6300,6400)
nu1,sw1 = getColumns('HF',['nu','sw'])
nu,coef = absorptionCoefficient_Lorentz(SourceTables='HF',HITRAN_units=False)
plt.plot(nu1,sw1,'.')
#plt.plot(nu,coef,'.')
plt.show()