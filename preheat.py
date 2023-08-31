import multiprocessing
import threading
import pandas as pd
import numpy as np
from math import log, isnan
from scipy.optimize import minimize
from IPython.display import display
from datetime import datetime, timedelta
#calculate time spent
import math
import time
import os
import sys

def LR_Cp_Curve(Cin,Cout):
    #calculate Cp of LR based on average inlet and outlet temperature
    return (0.0038 * (Cin + Cout)/2 + 1.7321)/86400
    
def Cp_Duty(Tin, Tout, Flow, Value, Type):
    'Do Q = mCpDT calculation. Value given as either Q or CP , Type to indicate the given value is Q or CP'
    
    mDT = Flow * abs(Tin-Tout)
    
    if Type =='Q':
        #calculate Cp using Q/(mDT)
        return Value/mDT if mDT > 0 else 0
    else:
        #calculate Q using mCpDT
        return Value*mDT   

def T_out(Tin,Flow,Cp,Q):
    'Calculate outlet temp based on single side data'
    return Q/(Cp*Flow) + Tin

def LMTD_Duty(Cin,Cout,Hin,Hout, Value, Type):
    'Do LMTD calculation. Value given as either Q or UA, Type to indicate the given value is Q or UA'

    if (Hin-Cout)/(Hout-Cin)==1 or (Hin-Cout)/(Hout-Cin)<=0:
        # Handle the case where the denominator is zero
        
        return 0
    else:
        # Compute the logarithm expression
        LMTD = ((Hin-Cout)-(Hout-Cin))/log((Hin-Cout)/(Hout-Cin))
        if Type == 'Q':
            #calculate UA using Q/LMTD
            return Value/LMTD
        else:
            #calculate Q using LMTD*UA
            return Value*LMTD

# Solve for Q when Co, Ho not available
def HXeq (Q, UA, Cflow, Ccp, Cin, Hflow, Hcp, Hin):
    if Cflow * Ccp <=0:
        Cout = Cin
    else:
        Cout = Q/ (Cflow * Ccp) + Cin
    if Hflow * Hcp <=0:
        Hout = Hin
    else:
        Hout = Hin - Q/ (Hflow * Hcp)
    
    #handle invalid LTMD
    if (Hin-Cout)/(Hout-Cin)==1 or (Hin-Cout)/(Hout-Cin)<=0:
        return 0, Cout, Hout
    else:
        
        return ((Hin-Cout)-(Hout-Cin))/log((Hin-Cout)/(Hout-Cin)) * UA - Q, Cout, Hout

def dxHXeq (Q, UA, Cflow, Ccp, Cin, Hflow, Hcp, Hin):
    delta = 0.001
    Y, Cout, Hout = HXeq (Q, UA, Cflow, Ccp, Cin, Hflow, Hcp, Hin)
    Y_1, Cout_1, Hout_1 = HXeq (Q+delta, UA, Cflow, Ccp, Cin, Hflow, Hcp, Hin)
    return (Y_1 - Y)/delta

def Qsolve(Q, UA, Cflow, Ccp, Cin, Hflow, Hcp, Hin):
    #solve for HX duty based on inlet temperature and flow only.
    max_tries = 100
    tries = 0
    err = 1 #initialise
    tol = 0.005
    while abs(err)>0.005 and tries <= max_tries:
        Qi = Q
        slope = dxHXeq (Qi, UA, Cflow, Ccp, Cin, Hflow, Hcp, Hin)
        err, Cout, Hout = HXeq (Qi, UA, Cflow, Ccp, Cin, Hflow, Hcp, Hin)
        
        if slope == 0:
            Q = Qi
        else:
            Q = Qi - err/slope
        tries +=1

    return Q, tries, Cout, Hout

#Set up optimisation problem

# Define the objective function and constraints
# x = [E1504_Cflow, E1504_Q, E1503_Cflow, E1503_Q, E1502_Cflow, E1502_Q ]

def objective_function(x, *args):
    #total duty
    #since function is minimisation, duty will be negative
    E1504_Q, E1504_UA, E1504_Ccp, E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin,\
    E1503_Q, E1503_UA, E1503_Ccp, E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin,\
    E1502_Q, E1502_UA, E1502_Ccp, E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin = args
    
    E1504_Qf, E1504_tries,E1504_Cout, E1504_Hout  = Qsolve(E1504_Q, E1504_UA, x[0], E1504_Ccp, E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin)
    #print(E1504_Qf)
    E1503_Qf, E1503_tries,E1503_Cout, E1503_Hout = Qsolve(E1503_Q, E1503_UA, x[1], E1503_Ccp, E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin)
    #print(E1503_Qf)
    E1502_Qf, E1502_tries,E1502_Cout, E1502_Hout = Qsolve(E1502_Q, E1502_UA, x[2], E1502_Ccp, E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin)
    #print(E1502_tries)
    
    return -(E1504_Qf + E1503_Qf + E1502_Qf)

def constraint_1(x,x0):
    #sum of all flow is same as initial guess
    #type ='eq'
    return (x[0]+x[1]+x[2]) - (x0[0]+x0[1]+x0[2]) # =0

def lower_bounds(x, args, previous_bound):

    E1504_Q, E1504_UA, E1504_Ccp, E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin,\
    E1503_Q, E1503_UA, E1503_Ccp, E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin,\
    E1502_Q, E1502_UA, E1502_Ccp, E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin = args
    
    ((E1504_lowflow_previous,E1504_highflow_previous),\
     (E1503_lowflow_previous,E1503_highflow_previous),\
     (E1502_lowflow_previous,E1502_highflow_previous)) = previous_bound
    
    # solve for duty
    E1504_Qf, E1504_tries,E1504_Cout, E1504_Hout = Qsolve(E1504_Q, E1504_UA, x[0], E1504_Ccp, E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin)
    
    E1503_Qf, E1503_tries,E1503_Cout, E1503_Hout = Qsolve(E1503_Q, E1503_UA, x[1], E1503_Ccp, E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin)
    
    E1502_Qf, E1502_tries,E1502_Cout, E1502_Hout = Qsolve(E1502_Q, E1502_UA, x[2], E1502_Ccp, E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin)
    
    
    #calculate boundaries of flow based on infeasible LMTD 
    #basically at every iteration a min flow is calculated to ensure that next guess stays within bounds
    E1504_lowflow = max(E1504_Qf/ (E1504_Ccp * ( E1504_Hin - E1504_Cin)) + 1,E1504_lowflow_previous)
    E1503_lowflow = max(E1503_Qf/ (E1503_Ccp * ( E1503_Hin - E1503_Cin)) + 1,E1503_lowflow_previous)
    E1502_lowflow = max(E1502_Qf/ (E1502_Ccp * ( E1502_Hin - E1502_Cin)) + 1,E1502_lowflow_previous)
    
    return ((E1504_lowflow,None),(E1503_lowflow,None),(E1502_lowflow,None))

## Calculate duty from cold flow.
# similar to objective function but provides more information

def final_duty(x, *args):
    #total duty
    #since function is minimisation, duty will be negative
    E1504_Q, E1504_UA, E1504_Ccp, E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin,\
    E1503_Q, E1503_UA, E1503_Ccp, E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin,\
    E1502_Q, E1502_UA, E1502_Ccp, E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin = args
    
    E1504_Qf, E1504_tries,E1504_Cout, E1504_Hout  = Qsolve(E1504_Q, E1504_UA, x[0], E1504_Ccp, E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin)
    #print(E1504_Qf)
    E1503_Qf, E1503_tries,E1503_Cout, E1503_Hout = Qsolve(E1503_Q, E1503_UA, x[1], E1503_Ccp, E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin)
    #print(E1503_Qf)
    E1502_Qf, E1502_tries,E1502_Cout, E1502_Hout = Qsolve(E1502_Q, E1502_UA, x[2], E1502_Ccp, E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin)
    #print(E1502_Qf)
    
    return E1504_Qf,E1504_Cout, E1504_Hout, E1503_Qf, E1503_Cout, E1503_Hout, E1502_Qf, E1502_Cout, E1502_Hout


def preheat_opt():
    e_old = ''
    errormsg_old = ''
    cached_stamp = 0
    csv_file = 'C:\\HV5_Python\\PREHEATIN.csv'
    while True:
        try:
            time.sleep(1)         
            stamp = os.stat(csv_file).st_mtime        
            if stamp != cached_stamp:
                cached_stamp = stamp
# File has changed, so do something..
            now = datetime.now()
            currentdatetime = now.strftime("%d/%m/%Y %H:%M")
            
            #tictoc and currenttime
            print(currentdatetime)
            tic = time.perf_counter()
            
            df = pd.read_csv('C:\\HV5_Python\\PREHEATIN.csv', encoding='utf-8')
            

            #calculate SR5 flow
            if df['15F132B.MV'][0] >= df['15F132A.MV'][0]:
                SR5 = df['15F132B.PV'][0]
            else:
                SR5 = df['15F132A.PV'][0]
            #define all parameters
            E1504_Cin = df['15T200.PV'][0]
            E1504_Cout = df['15T009.PV'][0]
            E1504_Cflow = df['15F003.PV'][0]
            E1504_Hin = df['15T123.PV'][0]
            E1504_Hout = df['15T010.PV'][0] # hi limit 253.4??? 
            E1504_Hflow = df['15F166.PV'][0]+df['15F046.PV'][0]-df['15F056.PV'][0]+df['15F048.PV'][0]+((df['15T125.PV'][0]*df['15F048.PV'][0])-(df['15T123.PV'][0]*df['15F048.PV'][0]))/(df['15T010.PV'][0]-df['15T123.PV'][0])

            E1503_Cin = df['15T200.PV'][0]
            E1503_Cout = df['15T012.PV'][0]
            E1503_Cflow = df['15F083.PV'][0]
            E1503_Hin = df['15T157.PV'][0]
            E1503_Hout = df['15T014.PV'][0]


            E1503_Hflow = df['15F004.PV'][0] + df['09F092.PV'][0] + df['09F086.PV'][0] - df['15F167.PV'][0] + SR5 - df['15F052.PV'][0] 

            E1502_Cin = df['15T200.PV'][0]
            E1502_Cout = df['15T001.PV'][0]
            E1502_Cflow = df['15F109.PV'][0]
            E1502_Hin =  df['15T133.PV'][0]
            E1502_Hout =  df['15T219.PV'][0]
            E1502_Hflow =  df['15F001.PV'][0]

            index+=1

            #E1504
            E1504_Ccp = LR_Cp_Curve(E1504_Cin,E1504_Cout)
            E1504_Q = Cp_Duty(E1504_Cin, E1504_Cout, E1504_Cflow, E1504_Ccp, 'cp')
            E1504_Hcp = Cp_Duty(E1504_Hin, E1504_Hout, E1504_Hflow, E1504_Q, 'Q')
            E1504_UA = LMTD_Duty(E1504_Cin,E1504_Cout,E1504_Hin,E1504_Hout, E1504_Q, 'Q')
            
            if round(E1504_Q,3) != round(Cp_Duty(E1504_Hin, E1504_Hout, E1504_Hflow, E1504_Hcp, 'cp'),3) or \
                round(E1504_Q,3) != round(LMTD_Duty(E1504_Cin,E1504_Cout,E1504_Hin,E1504_Hout, E1504_UA, 'UA'),3):
                E1504_Q_health = 0
                print('E1504 fail duty consistency', end ='\r')
            else:
                E1504_Q_health = 1


            #E1503
            E1503_Ccp = LR_Cp_Curve(E1503_Cin,E1503_Cout)
            E1503_Q = Cp_Duty(E1503_Cin, E1503_Cout, E1503_Cflow, E1503_Ccp, 'cp')
            E1503_Hcp = Cp_Duty(E1503_Hin, E1503_Hout, E1503_Hflow, E1503_Q, 'Q')
            E1503_UA = LMTD_Duty(E1503_Cin,E1503_Cout,E1503_Hin,E1503_Hout, E1503_Q, 'Q')

            if round(E1503_Q,3) != round(Cp_Duty(E1503_Hin, E1503_Hout, E1503_Hflow, E1503_Hcp, 'cp'),3)or \
            round(E1503_Q,3) != round(LMTD_Duty(E1503_Cin,E1503_Cout,E1503_Hin,E1503_Hout, E1503_UA, 'UA'),3):
                E1503_Q_health = 0
                print('E1503 fail duty consistency', end ='\r')
            else:
                E1503_Q_health = 1
                
            
            #E1502
            E1502_Ccp = LR_Cp_Curve(E1502_Cin,E1502_Cout)
            E1502_Q = Cp_Duty(E1502_Cin, E1502_Cout, E1502_Cflow, E1502_Ccp, 'cp')
            E1502_Hcp = Cp_Duty(E1502_Hin, E1502_Hout, E1502_Hflow, E1502_Q, 'Q')
            E1502_UA = LMTD_Duty(E1502_Cin,E1502_Cout,E1502_Hin,E1502_Hout, E1502_Q, 'Q')
            
            if round(E1502_Q,3) != round(Cp_Duty(E1502_Hin, E1502_Hout, E1502_Hflow, E1502_Hcp, 'cp'),3)or \
            round(E1502_Q,3) != round(LMTD_Duty(E1502_Cin,E1502_Cout,E1502_Hin,E1502_Hout, E1502_UA, 'UA'),3):
                E1502_Q_health = 0
                print('E1502 fail duty consistency', end ='\r')
            else:
                E1502_Q_health = 1
            
            

            # Define the initial guess for the variables
            x0 = [E1504_Cflow, E1503_Cflow, E1502_Cflow]


            # Define additional parameter for objective function
            args = (E1504_Q, E1504_UA, E1504_Ccp, E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin,
                   E1503_Q, E1503_UA, E1503_Ccp, E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin,
                   E1502_Q, E1502_UA, E1502_Ccp, E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin)

            # Define the constraints as a dictionary
            constraints = ({'type': 'eq', 'fun': constraint_1, 'args': (x0,)})

            # Define the bounds for the variables
            previous_bound = ((900,None),(2200,None),(350,None))

            max_iteration  = 100 ## change for loop to while loop when bound changes
            iteration = 0
            x_value = x0 
            duty_tot = sum([E1504_Q,E1503_Q,E1502_Q])
            
            
            #Skip optimisation if data health is not good
            # Data consistency checks duty calculated from Cold = Hot = LMTD calculation
            if E1504_Q_health*E1503_Q_health*E1502_Q_health == 0:
                tolerance = 0.0000001 # make tolerance small so loop doesnt start
                print('Skip optimisation due to duty inconsistency', end ='\r')
            else:
                tolerance = 1 # initialise loop
            
            while tolerance > 0.000001 and iteration < max_iteration:
                iteration += 1

                #calculate bound from LMTD constraints
                bounds = lower_bounds(x_value,args,previous_bound)

                # Use the SLSQP optimization algorithm to solve the problem
                result = minimize(objective_function, x_value ,args, method='SLSQP', bounds= bounds, constraints=constraints)
                x_value = result.x.tolist()

                #update bounds
                previous_bound = bounds

                #Calculate new duty based on optimisation
                E1504_Qf,E1504_Cout, E1504_Hout,\
                E1503_Qf, E1503_Cout, E1503_Hout,\
                E1502_Qf, E1502_Cout, E1502_Hout = final_duty(x_value, *args)

                #calculate tolerance

                tolerance = abs(duty_tot - sum([E1504_Qf,E1503_Qf,E1502_Qf]))
                duty_tot = sum([E1504_Qf,E1503_Qf,E1502_Qf])

                # calculate new update lR cp, and Qf, Cout and Hout
                args = (E1504_Qf, E1504_UA, LR_Cp_Curve(E1504_Cin,E1504_Cout), E1504_Cin, E1504_Hflow, E1504_Hcp, E1504_Hin,
                   E1503_Qf, E1503_UA, LR_Cp_Curve(E1503_Cin,E1503_Cout), E1503_Cin, E1503_Hflow, E1503_Hcp, E1503_Hin,
                   E1502_Qf, E1502_UA, LR_Cp_Curve(E1502_Cin,E1502_Cout), E1502_Cin, E1502_Hflow, E1502_Hcp, E1502_Hin)

            #output result
            duty_previous = sum([E1504_Q,E1503_Q,E1502_Q])
            improvement = duty_tot - duty_previous
            flow_change = [x - y for x,y in zip(x_value,x0)]
            T009_T012 = E1504_Cout - E1503_Cout
            T001 = E1502_Cout
           
            result_list = x_value + [T009_T012, T001, improvement]
            #set up list for result
            column_name =['E1504_Cflow', 'E1503_Cflow', 'E1502_Cflow',\
                'T009-T012','15T001.PV','duty_improvement']
            df_result =pd.DataFrame([result_list], columns = column_name)
            df_result.to_csv('C:\\HV5_Python\\PREHEATOUT.csv', encoding = 'utf-8', index = False)
            
            toc = time.perf_counter()
                    
            print(f"Runtime in {toc - tic:0.4f} seconds")
            print(df_result)
            
        except Exception as e:
            now = datetime.now()
            currentdatetime = now.strftime("%d/%m/%Y %H:%M")
            errormsg = 'Exception occurred: ' + str(e) + ' at ' + currentdatetime
            print(errormsg)

            if e_old == str(e):
                continue
            else:
                with open('C:\\HV5_Python\\Errorlog_preheat.txt','a') as f:
                    f.write('\n')
                    f.write(errormsg)
                e_old = str(e)
                continue

            
            
            

        


if __name__ == '__main__':
    # create and start the processes
    process1 = multiprocessing.Process(name='Preheat.calc', target = preheat_opt)

    process1.start()
    print(process1.name)

