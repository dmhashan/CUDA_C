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

