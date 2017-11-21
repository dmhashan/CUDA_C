echo "Welcome to performance analysis system"
echo "System need some information before analys your system"

printf "\nInformation for get more accurate result\n"
echo "-------------------------------------------------"
printf "Enter how many iterations you need to do : "
read user_i

printf "\nInformation for draw graph Execute Time against Matrix Size,\n"
echo "-------------------------------------------------"
printf "Block size for GPU calculations : "
read user_bsize
printf "Enter sizes of matrix : "
read user_msizes

printf "\nInformation for draw graph Execute Time against Block Size (For only GPU performance),\n"
echo "-------------------------------------------------"
printf "Constant Matrix Size for calculations : "
read user_msize
printf "Enter sizes of block : "
read user_bsizes

if [ "$user_i" -eq "$user_i" ] 2>/dev/null; then
  echo "$user_i" > ../data/user.dat
  echo "$user_bsize" >> ../data/user.dat
  echo "$user_msizes" >> ../data/user.dat
  echo "$user_msize" >> ../data/user.dat
  echo "$user_bsizes" >> ../data/user.dat
  clear;
  printf "* Successfully saved user data\n\n"
else
  cd ..
  sh run.sh
fi

