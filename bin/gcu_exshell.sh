#!/bin/bash

### User Input Gathering ###
user_file="../data/user.dat"
if [ -f "$user_file" ]
then
	line_count=`wc -l < $user_file`
	if [ "$line_count" -eq 5 ]; then
		itr=`sed -n 1p $user_file`
		bsize=`sed -n 2p $user_file`
		msizes=`sed -n 3p $user_file`
		msize=`sed -n 4p $user_file`
		bsizes=`sed -n 5p $user_file`
	else
		clear
		printf "* System haven't valid user data\n\n"
		sh user.sh
	fi
else
	clear
	printf "* System haven't valid user data\n\n"
	sh user.sh
fi

echo "********************************"
echo "GPU(Global) performance"
echo "********************************"
echo "Please wait . . ."

echo "p q" > ../data/dat_gxcu.dat

count=1
for x in $bsizes
do

sh gcu_bicode.sh $x
nvcc gcu.cu -o gcu

###calculate avg time
  total_time=0.0
  for k in `seq 1 1 $itr`
  do
	temp_time=`./gcu $msize `
	total_time=`echo $total_time + $temp_time | bc`
  done
  avg_time=`echo "scale = 10; $total_time/$itr" | bc`
###end calculate avg time

  indx[$count]=$x
  indy[$count]=$avg_time
  cordinate[$count]="(${indx[$count]},${indy[$count]})"

  echo "${indx[$count]} ${indy[$count]}" >> ../data/dat_gxcu.dat

  count=$((count+1))     
done
clear

echo "GPU(Global) performance"
echo "********************************"
echo ${cordinate[*]}
printf "\nSuccessfully write data into data_gxcu file\n"

echo "98) Main Menu"
echo "99) Exit"

printf "Select options using numbers, and press enter : "
read userinput

case $userinput in 
	98) clear; cd ..; sh run.sh;;
	99) clear; exit;;
	*) echo "invalid input";;
esac

