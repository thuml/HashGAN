#!/bin/bash

mkdir cifar10
scp 105:/home/caoyue/data/cifar10/train_.txt cifar10/train.txt &
scp 105:/home/caoyue/data/cifar10/parallel/database.txt cifar10/database.txt &
scp 105:/home/caoyue/data/cifar10/parallel/database_nolabel.txt cifar10/database_nolabel.txt & 
scp 105:/home/caoyue/data/cifar10/parallel/test.txt cifar10/test.txt &

mkdir nuswide_81
scp 105:/home/caoyue/data/nuswide_81/train_.txt nuswide_81/train.txt & 
scp 105:/home/caoyue/data/nuswide_81/test/database.txt nuswide_81/database.txt & 
scp 105:/home/caoyue/data/nuswide_81/test/database_nolabel.txt nuswide_81/database_nolabel.txt & 
scp 105:/home/caoyue/data/nuswide_81/test/test.txt nuswide_81/test.txt & 

mkdir coco
scp 105:/home/caoyue/data/coco/train_.txt coco/train.txt & 
scp 105:/home/caoyue/data/coco/test/database.txt coco/database.txt &
scp 105:/home/caoyue/data/coco/test/database_nolabel.txt coco/database_nolabel.txt &
scp 105:/home/caoyue/data/coco/test/test.txt coco/test.txt &
