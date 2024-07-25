#!/bin/bash
is=( 57 28 13 6 )
js=( 7 4 3 2 )
ks=( 64 32 16 8 )

for idx in "${!is[@]}"; do
	i=${is[$idx]}
	j=${js[$idx]}
	k=${ks[$idx]}
	echo $i $j $k
	#mkdir $k
	sed -e "s/n_train_value/$i/g" -e "s/n_val_value/$j/g" -e "s/train_val_total/$k/g" fine4class_gen.py > fine4class-$k'.py'
	sed "s/train_val_total/$k/g" fine4class_sub.sh > fine4class-$k'.sh'
	#cd $k
	qsub fine4class-$k'.sh'
	#cd $pwd
done
