##################################### lOAD ALL LIBS #################################
import PySimpleGUI as sg
from math import log, isnan, prod
import inspect
from subprocess import Popen
import time
import pandas as pd
import numpy as np   
from scipy.optimize import minimize
from win32com.client.dynamic import Dispatch
from PiPython import PiPython
from pathlib import Path
import sys
import os
#from tkinter import *
if getattr(sys, 'frozen', False):
    import pyi_splash
import ctypes

myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

root = os.path.dirname(__file__)
icon = os.path.join(root, 'images/app.ico')

################################################################################

############################# END OF LIBS ######################################    
sg.theme('Material1')
logwindow = sg.Multiline(key='logwin',size=(100, 20), font=('Courier', 10))
#print = logwindow.print
layout = [[sg.Text("Ensure that input_template.csv is updated")], [sg.Button("RUN")],
          [sg.Text('Select start and end date for PI historical data')],
          [sg.CalendarButton('Start date',target ='start_date',format='%d/%m/%Y'),sg.In(key='start_date', enable_events=True, visible=True) ],
          [sg.CalendarButton('End date',target ='end_date',format='%d/%m/%Y'), sg.In(key='end_date', enable_events=True, visible=True) ],
          [sg.Button("EXTRACT")],[sg.Button("OPTIMIZE")],[logwindow]]

# Create the window
window = sg.Window("Heat Pinch Opt", layout, icon=icon)

if getattr(sys, 'frozen', False):
    pyi_splash.close()
# Create an event loop
while True:
    event, values = window.read(timeout=100)
    # End program if user closes window or
    #####check if input file exist#######
    path_to_file = 'input_template.csv'
    path = Path(path_to_file)

    if not path.is_file():
        logwindow.print(f'The file {path_to_file} does not exist, creating file')
        new_inputdf = pd.DataFrame(columns = ['Train', 'Series', 'Exchanger', 'Cold_flow','Cold_in','Cold_out','Hot_flow','Hot_in','Hot_out','min_flow','max_flow','CP_gain','CP_intercept'])
        new_inputdf.to_csv('input_template.csv',encoding='utf-8', index =False)
        logwindow.print(f'The file {path_to_file} exists, opening file')
        p = Popen('input_template.csv', shell=True)
        
    #####################################

    if event == sg.WIN_CLOSED:
        break
        
     # presses the RUN button to inspect CSV template   
    if event == "RUN":
        #Read template file
        template = pd.read_csv('input_template.csv', encoding ='utf-8')
        #sort by Train then by Series so that Exchangers calculated are run in the correct order
        template = template.sort_values(['Train', 'Series'])
        Position_df = template[['Train','Series','Exchanger']]
        limit_df = template[['Train','min_flow','max_flow']]
        limit_df = limit_df.groupby('Train').agg({'min_flow': 'min', 'max_flow': 'max'}).reset_index()
        template = template.drop(columns=['Train','Series','min_flow','max_flow','CP_gain','CP_intercept'])
        template.set_index(template.columns[0],inplace=True)
        
        logwindow.print(Position_df)
        logwindow.print(limit_df)
        logwindow.print(template)
        
    
    # press extract button to pull PI data    
    if event == "EXTRACT" and (values['start_date'] != '' or values['end_date'] != ''):
        
        ## Start progress bar before download
        prog_window = sg.Window('PI download',  [[sg.Text("Downloading PI data")],
           [sg.ProgressBar(100, orientation='h', expand_x=True, size=(20, 20),  key='-PBAR-')],
            ], size=(715, 150))
        
        event2, values2 = prog_window.read(timeout=100)
        
        prog_window['-PBAR-'].update(current_count=10 + 1)
        time.sleep(1)
        
        # Function to replace negative values with zero
        def replace_negatives(x):
            if x < 0:
                return np.nan
            else:
                return x

        #Read template file
        template = pd.read_csv('input_template.csv', encoding ='utf-8')
        #sort by Train then by Series so that Exchangers calculated are run in the correct order
        template = template.sort_values(['Train', 'Series'])
        Position_df = template[['Train','Series','Exchanger']]
        limit_df = template[['Train','min_flow','max_flow']]
        limit_df = limit_df.groupby('Train').agg({'min_flow': 'min', 'max_flow': 'max'}).reset_index()
        template = template.drop(columns=['Train','Series','min_flow','max_flow','CP_gain','CP_intercept'])
        template.set_index(template.columns[0],inplace=True)
        

        
        prog_window['-PBAR-'].update(current_count=20 + 1)
        time.sleep(1)
        
        #Extract pi data from the template
        taglist = list(set(template.values.flatten().tolist()))
        PI = PiPython.PiServer('DSAPPICOLL')
        ST = values['start_date'] + ' 00:00 AM'
        ET = values['end_date'] + ' 00:00 AM'
        

        prog_window['-PBAR-'].update(current_count=50 + 1)

        df = PI.PItoDF(taglist, ST, ET, Interval = '1m', Timeweighted = True ,Batchsize = 1)
        # Remove the 'date' column
        
        prog_window['-PBAR-'].update(current_count=90 + 1)
        time.sleep(1)
        
        date_column = df.pop('date')
        df = df.applymap(replace_negatives)
        df['date'] = date_column
        prog_window['-PBAR-'].update(current_count=99 + 1)
        time.sleep(1)
        df.to_csv('data_buffer.csv', encoding ='utf-8', index =False)
        
        prog_window.close()
        logwindow.print('PI data extracted to buffer.csv')
    elif event =='EXTRACT':
        logwindow.print('Date not selected yet')
        
    if event == "OPTIMIZE":
        logwindow.print('Loading functions')
                ################################# LOAD ALL FUNCTIONS ###########################
        def LR_Cp_Curve(Cin,Cout,CP_gain,CP_intercept):
            #calculate Cp of LR based on average inlet and outlet temperature
            #crude (0.0041* ((Cin + Cout)/2) + 1.8421)/86400
            #LR (0.0038 * (Cin + Cout)/2 + 1.7321)/86400
            return (CP_gain * (Cin + Cout)/2 + CP_intercept)/86400

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

        def data_transfer(var, ref_prop, ref_tag, variables):
        #Create a data transfer function
        # data transfer function will get string e.g. E2501_Cold_flow and look into ref_prop to see the value (25F111.PV) 
        # Using this value to look into ref_tag for all the properties related. e.g E2501_Cold_flow, E2502_Cold_fLow
        # Update the value in variables for these 2 properties
        # data transfer to be used in every steps of calculations why optimising    

            # Get the caller's frame
            frame = inspect.currentframe().f_back

            # Iterate over the frame's local variables and check if any variable has the same id as the input var
            for name, value in frame.f_locals.items():
                if id(value) == id(var):
                    input_str =  name
            # Iterate over the frame's global variables and check if any variable has the same id as the input var
            for name, value in frame.f_globals.items():
                if id(value) == id(var):
                    input_str = name


            # Look up the PI tags in ref_prop
            PI_tag = ref_prop.get(input_str)


            if PI_tag is not None:
                # Look up the related properties in ref_tag
                related_properties = ref_tag.get(PI_tag)

                # Update the all values of the same PI tags in variables
                for prop in related_properties:
                    variables[prop] = var
        #Set up optimisation problem

        def objective_function(x, *args):
            #total duty
            #since function is minimisation, duty will be negative
            # x is solution based on number of trains.
            # assign x as based on train number

            variables, Position_df, ref_prop, ref_tag = args
            #number of exchanger = number of loops to run
            no_hx = len(Position_df)
            Total_duty = 0
            for index in range(len(Position_df)):
                HX = Position_df['Exchanger'][index]
                train_no = Position_df['Train'][index]-1 #train 1 will be on position 0 of x

                globals()[f'{HX}_Q'], globals()[f'{HX}_tries'], globals()[f'{HX}_Cold_out'], globals()[f'{HX}_Hot_out']\
                = Qsolve(variables[f'{HX}_Q'], variables[f'{HX}_UA'], x[train_no], variables[f'{HX}_Ccp'],\
                         variables[f'{HX}_Cold_in'],variables[f'{HX}_Hot_flow'],variables[f'{HX}_Hcp'],variables[f'{HX}_Hot_in'])


                Total_duty -= globals()[f'{HX}_Q']


            return Total_duty

        def constraint_1(x,x0):
            #sum of all flow is same as initial guess
            #type ='eq'
            return sum(x) - sum(x0) # =0


        def lower_bounds(x, variables, previous_bound, Position_df):
            #recalculate new bounds
            # x is solution based on number of trains.
            # assign x as based on train number
            #number of exchanger = number of loops to run
            no_hx = len(Position_df)
            for index in range(len(Position_df)):
                HX = Position_df['Exchanger'][index]
                train_no = Position_df['Train'][index]-1 #train 1 will be on position 0 of x

                globals()[f'{HX}_Q'], globals()[f'{HX}_tries'], globals()[f'{HX}_Cold_out'], globals()[f'{HX}_Hot_out']\
                = Qsolve(variables[f'{HX}_Q'], variables[f'{HX}_UA'], x[train_no], variables[f'{HX}_Ccp'],\
                         variables[f'{HX}_Cold_in'],variables[f'{HX}_Hot_flow'],variables[f'{HX}_Hcp'],variables[f'{HX}_Hot_in'])


                #calculate boundaries of flow based on infeasible LMTD 
                #basically at every iteration a min flow is calculated to ensure that next guess stays within bounds
                new_bound = list(previous_bound)

                new_bound[train_no] = (max(globals()[f'{HX}_Q']/ (variables[f'{HX}_Ccp'] * ( variables[f'{HX}_Hot_in']\
                                            - variables[f'{HX}_Cold_in']  )) + 1 ,new_bound[train_no][0]),new_bound[train_no][1])

            return tuple(new_bound)
        ############################## End of Functions ###################################
        logwindow.print('Loaded functions')
        #Read template file
        template = pd.read_csv('input_template.csv', encoding ='utf-8')
        #sort by Train then by Series so that Exchangers calculated are run in the correct order
        template = template.sort_values(['Train', 'Series'])
        Position_df = template[['Train','Series','Exchanger']]
        limit_df = template[['Train','min_flow','max_flow']]
        limit_df = limit_df.groupby('Train').agg({'min_flow': 'min', 'max_flow': 'max'}).reset_index()
        
        CP_gain = template['CP_gain'][0]
        CP_intercept = template['CP_intercept'][0]
        
        if isnan(CP_gain):
            #use default value of CP_gain
            CP_gain = 0.004
            logwindow.print('No CP_gain provided, default to 0.004 for crude')
    
        
        if isnan(CP_intercept):
            #use default value of CP_intercept
            CP_intercept = 1.8
            logwindow.print('No CP_intercept provided, default to 1.8 for crude')
        
        template = template.drop(columns=['Train','Series','min_flow','max_flow','CP_gain','CP_intercept'])
        template.set_index(template.columns[0],inplace=True)
        
 
        df_full = pd.read_csv('data_buffer.csv', encoding ='utf-8') #read CSV file with PI values
        length = df_full.shape[0]

        #ref_prop key is e.g. E1501_Cold_flow with value being the PI tag
        ref_prop = {}
        #ref_tag key is PI tag with value being the properties e.g E2501_Cold_flow
        ref_tag = {}
        #variable key is the properties e.g. E2501_Cold_flow, with the value being the numerical flow rate aka variables.
        variables = {}


        ########## create columns names for new dataframe###########################
        list_exchanger = Position_df[Position_df['Series']==1]['Exchanger'].values.tolist()
        column_names = []
        for item in list_exchanger: 
            column_names.append(item + '_Cold_flow') 

        column_names.append('duty_total')
        column_names.append('duty_previous')
        column_names.append('duty_improvement')
        for item in list_exchanger: 
            column_names.append(item + '_delta') 

        df_result =pd.DataFrame(columns = column_names)
        #################################################################################
        
        #iterate df_full

        for df_index in range(length):
            
            logwindow.print('Running Calculations for: {} out of {} data'.format(df_index+1,length))
            window.refresh()
            df = df_full.loc[[df_index]].reset_index()


            #Creating dictionary for ref_prop and variables
            for row_label, row in template.iterrows():
                for col_label in template.columns:
                    cell_value = row[col_label]
                    variable_name = f'{row_label}_{col_label}'.replace('-', '_').replace(' ', '_')
                    PI_value = df[cell_value][0]
                    variables[variable_name] = PI_value
                    ref_prop[variable_name] = cell_value

                #for each train calculate the following: Ccp/Q/Hcp/UA
                variables[f'{row_label}_Ccp'] = LR_Cp_Curve(variables[f'{row_label}_Cold_in'],variables[f'{row_label}_Cold_out'],CP_gain,CP_intercept)
                variables[f'{row_label}_Q'] = Cp_Duty(variables[f'{row_label}_Cold_in'], variables[f'{row_label}_Cold_out'], variables[f'{row_label}_Cold_flow'], variables[f'{row_label}_Ccp'], 'cp')
                variables[f'{row_label}_Hcp'] = Cp_Duty(variables[f'{row_label}_Hot_in'], variables[f'{row_label}_Hot_out'], variables[f'{row_label}_Hot_flow'], variables[f'{row_label}_Q'], 'Q')
                variables[f'{row_label}_UA'] = LMTD_Duty(variables[f'{row_label}_Cold_in'],variables[f'{row_label}_Cold_out'],variables[f'{row_label}_Hot_in'],variables[f'{row_label}_Hot_out'], variables[f'{row_label}_Q'], 'Q')
                # check Consistency
                if round(variables[f'{row_label}_Q'],3) != round(Cp_Duty(variables[f'{row_label}_Hot_in'], variables[f'{row_label}_Hot_out'], variables[f'{row_label}_Hot_flow'], variables[f'{row_label}_Hcp'], 'cp'),3) or \
                round(variables[f'{row_label}_Q'],3) != round(LMTD_Duty(variables[f'{row_label}_Cold_in'],variables[f'{row_label}_Cold_out'],variables[f'{row_label}_Hot_in'],variables[f'{row_label}_Hot_out'], variables[f'{row_label}_UA'], 'UA'),3):
                    variables[f'{row_label}_Q_health'] = 0
                    print(row_label + 'fail duty consistency') #, end ='\r'
                else:
                    variables[f'{row_label}_Q_health'] = 1

            #Creating ref_tag
            for key, value in ref_prop.items():
                # Check if the value is already in the swapped dictionary
                if value in ref_tag:
                    # Append the current key to the list of keys for the value
                    ref_tag[value].append(key)
                else:
                    # Create a new list with the current key as the only value
                    ref_tag[value] = [key]



            # Define the initial guess for the variables
            # optimising the cold flow
            # identifying the target variables to change
            # assumption is that cold flow for each train
            # pick up all the series 1 exchangers which represent the first exchangers of each train
            list_exchanger = Position_df[Position_df['Series']==1]['Exchanger'].values.tolist()
            #get the corresponding cold flow values from the variables storage
            x0 = [variables[f'{item}_Cold_flow'] for item in list_exchanger]


            # Define the constraints as a dictionary
            constraints = ({'type': 'eq', 'fun': constraint_1, 'args': (x0,)})


            # Define the bounds for the variables
            # starting bounds are limited at (0,None)

            #previous_bound = ((350,None),)*len(x0)
            previous_bound = tuple(
                tuple(None if isinstance(i, float) and isnan(i) else i for i in t)
                    for t in list(zip(limit_df['min_flow'], limit_df['max_flow']))
)

            max_iteration  = 100 ## change for loop to while loop when bound changes
            iteration = 0
            x_value = x0 


            # Define additional parameter for objective function
            args = (variables, Position_df, ref_prop, ref_tag)

            filtered_Q = {key: value for key, value in variables.items() if key.endswith('_Q')}
            duty_tot =  sum(filtered_Q.values())
            duty_previous = duty_tot

            #Skip optimisation if data health is not good
            # Data consistency checks duty calculated from Cold = Hot = LMTD calculation
            filtered_Q_health = {key: value for key, value in variables.items() if key.endswith('_Q_health')}


            if prod(filtered_Q_health.values()) == 0:
                tolerance = 0.0000001 # make tolerance small so loop doesnt start
                logwindow.print('Skip optimisation due to duty inconsistency', end ='\r')
            else:
                tolerance = 1 # initialise loop

            while tolerance > 0.000001 and iteration < max_iteration:
                iteration += 1

                #calculate bound from LMTD constraints
                bounds = lower_bounds(x_value, variables, previous_bound, Position_df)

                # Use the SLSQP optimization algorithm to solve the problem
                result = minimize(objective_function, x_value ,args, method='SLSQP', bounds= bounds, constraints=constraints)
                x_value = result.x.tolist()


                #update bounds
                previous_bound = bounds

                #Update variables with newly optimised x_value
                #number of exchanger = number of loops to run
                no_hx = len(Position_df)

                for index in range(len(Position_df)):
                    HX = Position_df['Exchanger'][index]
                    train_no = Position_df['Train'][index]-1 #train 1 will be on position 0 of x
                    globals()[f'{HX}_Q'], globals()[f'{HX}_tries'], globals()[f'{HX}_Cold_out'], globals()[f'{HX}_Hot_out']\
                    = Qsolve(variables[f'{HX}_Q'], variables[f'{HX}_UA'], x_value[train_no], variables[f'{HX}_Ccp'],\
                             variables[f'{HX}_Cold_in'],variables[f'{HX}_Hot_flow'],variables[f'{HX}_Hcp'],variables[f'{HX}_Hot_in'])


                    variables[f'{HX}_Q'] = globals()[f'{HX}_Q']
                    globals()[f'{HX}_Cold_flow'] = x_value[train_no]
                    data_transfer(globals()[f'{HX}_Cold_flow'], ref_prop, ref_tag, variables)
                    data_transfer(globals()[f'{HX}_Cold_out'], ref_prop, ref_tag, variables)
                    data_transfer(globals()[f'{HX}_Hot_out'], ref_prop, ref_tag, variables)     


                # update args with newly calculated values stored in variables for next loop
                args = (variables, Position_df, ref_prop, ref_tag)

                filtered_Q = {key: value for key, value in variables.items() if key.endswith('_Q')}
                duty_tot_new =  sum(filtered_Q.values())


                #calculate tolerance
                tolerance = abs(duty_tot - duty_tot_new)
                duty_tot = duty_tot_new

            #output result
            improvement = duty_tot - duty_previous
            flow_change = [x - y for x,y in zip(x_value,x0)]



            result_list = x_value +[duty_tot,duty_previous,improvement] + flow_change
            #set up list for result
            df_new_result =pd.DataFrame([result_list], columns = column_names)
            df_result = df_result.append(df_new_result, ignore_index=True)

            
#########################################################################
        df_result.to_csv('Optimisation_results.csv', encoding ='utf-8', index =False)
        
        logwindow.print('Opening results in excel')
        p = Popen('Optimisation_results.csv', shell=True)

        
window.close()       