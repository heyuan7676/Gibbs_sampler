#!/bin/bash

cd /home/yhe23/suchi/hw3b


iter_max=1100
burn_in=1000


### 5.1

## 1
./collapsed-sampler input-train.txt input-test.txt collapsed_1 25 0.5 0.1 0.01 ${iter_max} ${burn_in} > collapsed_1_output
./collapsed-sampler input-train.txt input-test.txt collapsed_2 25 0.5 0.1 0.01 ${iter_max} ${burn_in} > collapsed_2_output
./collapsed-sampler input-train.txt input-test.txt collapsed_3 25 0.5 0.1 0.01 ${iter_max} ${burn_in} > collapsed_3_output

## 2
./blocked-sampler input-train.txt input-test.txt blocked 25 0.5 0.1 0.01 ${iter_max} ${burn_in} > blocked_output


## 4
for topic in {10,20,30,40,50}
do
    echo $topic
    ./collapsed-sampler input-train.txt input-test.txt collapsed_topic_${topic} ${topic} 0.5 0.1 0.01 ${iter_max} ${burn_in} > collapsed_topic_${topic}_output
done

## 5
for lmda in 0 0.25 0.5 0.75 1
do
    echo $lmda
    ./collapsed-sampler input-train.txt input-test.txt collapsed_lmda_${lmda} 25 ${lmda} 0.1 0.01 ${iter_max} ${burn_in} > collapsed_lmda_${lmda}_output
done


## 6
for alpha in 0.001 10.0
do
    ./collapsed-sampler input-train.txt input-test.txt collapsed_alpha_${alpha} 25 .5 ${alpha} 0.01 ${iter_max} ${burn_in} > collapsed_alpha_${alpha}_output
done

for beta in 0.001 10.0
do
    ./collapsed-sampler input-train.txt input-test.txt collapsed_beta_${beta} 25 .5 0.1 ${beta} ${iter_max} ${burn_in} > collapsed_beta_${beta}_output
done
