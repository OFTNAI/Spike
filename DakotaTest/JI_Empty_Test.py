import sys
import time
import subprocess
import os;

objectiveFunc = 1; #1:inforAnalysis, 2:PI, 3:both

experimentName="dakota_visnet_BO_imgs5revised_5connections_smallGmax_trans"
imageFolder = "visnet_BO_imgs5revised_train_mod_two";#"visnet_BO_imgs5revised_train_mod_two"
experimentName = experimentName + time.strftime("_%Y.%m.%d_%H.%M.%S", time.gmtime());
print str(sys.argv);

# #set params
outputFile = sys.argv[1];
simulation_index = sys.argv[2];
# gmax =  float(sys.argv[2]);
# tau_syn_const = float(sys.argv[3]);


# f = open(os.path.split(os.path.realpath(__file__))[0] +"/Results/"+experimentName+"/params.txt","w");
# f.write(str(sys.argv));
# f.close();
# source = os.path.split(os.path.realpath(__file__))[0] +"/Dakota/params.in";
# destination = os.path.split(os.path.realpath(__file__))[0] +"/Results/"+experimentName+"/params.in"
# subprocess.call("cp " + source + " " + destination, shell=True);

# #copy the results into dakota folder
# if objectiveFunc==1:
#     source = os.path.split(os.path.realpath(__file__))[0] +"/Results/"+experimentName+"/performance.txt";
# elif objectiveFunc==2:
#     source = os.path.split(os.path.realpath(__file__))[0] +"/Results/"+experimentName+"/PI_improvement.txt";
# elif objectiveFunc==3:
#     source = os.path.split(os.path.realpath(__file__))[0] +"/Results/"+experimentName+"/infoAndPI.txt";


    
# destination = outputFile;
# subprocess.call("cp " + "0.4" + " " + destination, shell=True);

#
f = open(os.path.split(os.path.realpath(__file__))[0] + "/workdir." + simulation_index + "/" + outputFile,"w");
f.write(simulation_index);
f.close();