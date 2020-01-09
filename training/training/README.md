# Model Training (TF 1.14)
Use the notebook "Train Net" for Training. Then the Keras file will be exported. 
In order to successfully export the frozen graph(.pb) you need to load and store the Keras seperatly.(Little bit hacky)

As trainingsdata i have used the GTSRB(http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)


## Conversion from .pb to uff
Then use the Nvidia Tool to create an .uff file from frozen graph(.pb) 
-> https://devtalk.nvidia.com/default/topic/1025246/jetson-tx2/where-is-convert-to-uff/
