# Run main in a loop

import os
import main
import json
import time
from utils.aggregator import aggregate_all
latent_dims = [2,5,10,20,30,50,75,100]
epochs = [50,200,500,1000]


#thresholds = [2000,5000,10000,15000,20000]
#num_neurons = [100,75,50,30,20,10,5]


#David's run
#thresholds = [2000,5000,10000]#,15000,20000]
#num_neurons = [100,75,50]#,30,20,10,5]


#Hagai's run
thresholds = [100, 2000,5000,10000]#,15000,20000]
num_neurons = [30,20,10,5]


#Asia's run
#thresholds = [15000,20000]
#num_neurons = [100,75,50,30,20,10,5]

print
with open('results.txt',"w",0) as r:
    r.write("30 randomizations, cross_ent\n")
    r.write("threshold\tnum_neurons\tlatent_dim\tnum_epochs\tAverage_VAE\tVar_VAE\tAverage_PCA\tVar_PCA\n")
    rows_output=[]
    for d in latent_dims:
        for e in epochs:
            for t in thresholds:
                for n in num_neurons:
                    app_config_json = json.load(open('config/app_config.json'))
                    app_config_json['var_th_index'] = t
                    app_config_json['number_of_neurons'] = n
                    app_config_json['latent_dim'] = d
                    app_config_json['num_of_epochs'] = e
                    json.dump(app_config_json, open('config/app_config.json','w+'))

                    results = []
                    ##result proc
                    results = main.run(var_th_index=t,number_of_neurons=n, latent_dim=d, num_of_epochs=e)

        cur_row = str(t)+"\t"+str(n)+"\t"+str(d)+"\t"+str(e)+"\t"
        for cur_line in results:
            psplit = cur_line[1].strip().split(",")
            method = psplit[0].split(":")[0].split("_")[0]
            avg = psplit[0].split(":")[1]
            var = psplit[1].split(":")[1]
            cur_row += (str(avg)+'\t'+str(var)+'\t')

        open(os.path.join("results_{}_{}_{}_{}.txt".format(d,e,t,n)),'w+').write(cur_row)


    aggregate_all(latent_dims, epochs, thresholds, num_neurons,time.time())




