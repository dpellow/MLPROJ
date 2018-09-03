# Run main in a loop

import os
import constants 
import main

#thresholds = [2000,5000,10000,15000,20000]
#num_neurons = [100,75,50,30,20,10,5]


#David's run
#thresholds = [2000,5000,10000]#,15000,20000]
#num_neurons = [100,75,50]#,30,20,10,5]

#Hagai's run
thresholds = [2000,5000,10000]#,15000,20000]
num_neurons = [30,20,10,5]

#Asia's run
#thresholds = [15000,20000]
#num_neurons = [100,75,50,30,20,10,5]

latent_dims = [2,5,10,20,30,50,75,100]
epochs = [50,200,500,1000]


with open('results.txt',"w",0) as r:
 r.write("30 randomizations, cross_ent\n")
 r.write("threshold\tnum_neurons\tlatent_dim\tnum_epochs\tAverage_VAE\tVar_VAE\tAverage_PCA\tVar_PCA\n")
 for t in thresholds:
  for n in num_neurons:
    for d in latent_dims:
      for e in epochs:
        with open('config/app_config.json') as f:
          lines = f.readlines()
        with open('config/app_config.json','w') as o:
          for l in lines:
            if 'var_th_index' in l:
              o.write('  "var_th_index" : ' + str(t) + ',\n')
            elif 'number_of_neurons' in l:
              o.write('  "number_of_neurons" : ' + str(n) + ',\n')
            elif 'latent_dim' in l:
              o.write('  "latent_dim" : ' + str(d) + ',\n')
            elif 'num_of_epochs' in l:
              o.write('  "num_of_epochs" : ' + str(e) + ',\n')
            else:
              o.write(l)

        main.run()
        with open(os.path.join(constants.OUTPUT_GLOBAL_DIR, 'PANCAN.txt')) as f:
          lines = f.readlines()
        vsplit = lines[0].strip().split(",")
        vae_avg = vsplit[0].split(":")[1]
        vae_var = vsplit[1].split(":")[1]
        
        psplit = lines[1].strip().split(",")
        pca_avg = psplit[0].split(":")[1]
        pca_var = psplit[1].split(":")[1]
        r.write(str(t)+"\t"+str(n)+"\t"+str(d)+"\t"+str(e)+"\t"+str(vae_avg)+'\t'+str(vae_var)+'\t'+str(pca_avg)+'\t'+str(pca_var)+'\n')

 
