# Run main in a loop

import os
import main
import json
import time
from utils.aggregator import aggregate_all
latent_dims = [2]# ,5,10,20,30,50,75,100]
epochs = [2]# , 50,200,500,1000]


#thresholds = [2000,5000,10000,15000,20000]
#num_neurons = [100,75,50,30,20,10,5]


#David's run
#thresholds = [2000,5000,10000]#,15000,20000]
#num_neurons = [100,75,50]#,30,20,10,5]


#Hagai's run
thresholds = [50] # , 2000,5000,10000]#,15000,20000]
num_neurons = [3,4]# ,20,10,5]


#Asia's run
#thresholds = [15000,20000]
#num_neurons = [100,75,50,30,20,10,5]

clear_all(latent_dims, epochs, thresholds, num_neurons,time.time())


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
                    json.dump(app_config_json, open('config/app_config.json','w+'), indent=4, sort_keys=True)

                    results = []
                    ##result proc
                    results = main.run(var_th_index=t,number_of_neurons=n, latent_dim=d, num_of_epochs=e)

        	    cur_row = str(t)+"\t"+str(n)+"\t"+str(d)+"\t"+str(e)+"\t"
        	    for cur_line in results:
			print cur_line
            		method = cur_line['type']
            		avg = cur_line['avg']
            		var = cur_line['var']
            		cur_row += (method +'\t'+str(avg)+'\t'+str(var)+'\t')

        		open(os.path.join("results_{}_{}_{}_{}.txt".format(d,e,t,n)),'w+').write(cur_row)


aggregate_all(latent_dims, epochs, thresholds, num_neurons,time.time())




