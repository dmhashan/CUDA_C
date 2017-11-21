clear

### User Input Gathering ###
user_file="data/user.dat"
if [ -f "$user_file" ]
then
	line_count=`wc -l < $user_file`
	if [ "$line_count" -eq 5 ]; then
		echo "User Details"
		echo "-----------------------------------------"
		echo "# of iterations `sed -n 1p $user_file`"
		echo "Matrix sizes `sed -n 3p $user_file`"
		echo "Block sizes `sed -n 5p $user_file`"
		echo "Constant block size : `sed -n 2p $user_file`"
		echo "Constant matrix size : `sed -n 4p $user_file`"
		echo ""
		echo ""
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


echo "****************************************"
echo "Welcome to CPU GPU performance analycer"
echo "****************************************"
echo ""
echo "1) Make data file for CPU processing"
echo "2) Make data file for GPU(Global) processing depends on matrix size"
echo "3) Make data file for GPU(Shared) processing depends on matrix size"
echo "4) Make data file for GPU(Global) processing depends on block size"
echo "5) Make data file for GPU(Shared) processing depends on block size"
echo "6) Make report pdf"
echo "7) Change user data"
echo "99) Exit"

printf "\nSelect options using numbers, and press enter : "
read userinput

case $userinput in 
	1) clear; cd bin; bash c_shell.sh;;
	2) clear; cd bin; bash gcu_shell.sh;;
	3) clear; cd bin; bash scu_shell.sh;;
	4) clear; cd bin; bash gcu_exshell.sh;;
	5) clear; cd bin; bash scu_exshell.sh;;
	6) clear; cd bin; bash pdf.sh;;
	7) clear; cd bin; bash user.sh;;
	99) clear; exit;;
	*) echo "invalid input";;
esac
