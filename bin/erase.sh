printf "\nSuccessfully erase/clean data files\n"

echo "98) Main Menu"
echo "99) Exit"

printf "Select options using numbers, and press enter : "
read userinput

case $userinput in 
	98) clear; cd ..; sh run.sh;;
	99) clear; exit;;
	*) echo "invalid input";;
esac

