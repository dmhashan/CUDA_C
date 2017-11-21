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


cat > ../data/report.tex << EOF1
	
	
	\documentclass[12pt]{report}
	\usepackage[utf8]{inputenc}
	\usepackage{graphicx}
	\usepackage{tikz}
	\usepackage{pgfplots}
	\usepackage{textcomp}
	\usepackage{siunitx}
	\usepackage{pgfplotstable}
	\usepackage{booktabs}
	\usepackage{array}
	\usepackage{colortbl}

	\graphicspath{ {images/} }
	\pgfplotsset{width=10cm}
	
	\pgfplotstableset{% global config, for example in the preamble
	  every head row/.style={before row=\toprule,after row=\midrule},
	  every last row/.style={after row=\bottomrule},
	  fixed,precision=5,
	}

	\title{
		{Parallel Distributed Computing}\\
		{\large CUDA Programming Assignment 01}\\
		{\includegraphics[scale=0.5]{logo.jpg}}
	}
	\author{D. M. H. Hirosh ( UWU/CST/14/0014 )}
	\date{\today}



	\begin{document}
	 
		\begin{titlepage}

		\maketitle
		\end{titlepage}

	\tableofcontents{}

		\chapter{Graphs and Tables to compare performance differences agains matrix size}
		\input{chapters/chapter01}

		\chapter{Graphs and Tables to compare performance differences agains threads per block}
		\input{chapters/chapter02}

		\chapter{Small descriptions about graphs}
		\input{chapters/chapter03}

		\chapter{Conclusions}
		\input{chapters/chapter04}

		\bibliographystyle{unsrt}
		\bibliography{sample}
		\input{chapters/chapter05}
	 
	\end{document}

EOF1

cat > ../data/chapters/chapter01.tex << EOF2
\pagebreak
\pgfplotstableread{../data/dat_c.dat}\cdat
\pgfplotstableread{../data/dat_gcu.dat}\gcudat		
\pgfplotstableread{../data/dat_scu.dat}\scudat		

\section{CPU performance agains matrix size}
		
		\pgfplotstabletypeset[
		  columns/p/.style={column name=Matrix Size},
		  columns/q/.style={column name=Executing time},
		]{\cdat}
\bigbreak
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains size of matrix)},
			xlabel={Size of the matrix},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=blue,mark=x]table {\cdat};
		\end{axis}
		\end{tikzpicture}

		 
\section{GPU(Global) performance agains matrix size}
		
		\pgfplotstabletypeset[
		  columns/p/.style={column name=Matrix Size},
		  columns/q/.style={column name=Executing time},
		]{\gcudat}
\bigbreak		
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains size of matrix)},
			xlabel={Size of the matrix},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=blue,mark=x]table {\gcudat};
		\end{axis}
		\end{tikzpicture}

		 
\section{GPU(Shared) performance agains matrix size}
		
		\pgfplotstabletypeset[
		  columns/p/.style={column name=Matrix Size},
		  columns/q/.style={column name=Executing time},
		]{\scudat}
\bigbreak		
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains size of matrix)},
			xlabel={Size of the matrix},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=blue,mark=x]table {\scudat};
		\end{axis}
		\end{tikzpicture}

\section{CPU and GPU(Global/Shared) performance differences agains size of matrix}
		
		\pgfplotstablecreatecol[copy column from table={\gcudat}{[index] 1}] {r} {\cdat}
		\pgfplotstablecreatecol[copy column from table={\scudat}{[index] 1}] {s} {\cdat}

		\pgfplotstabletypeset[
		  columns/p/.style={column name=Matrix Size},
		  columns/q/.style={column name=CPU},
		  columns/r/.style={column name=GPU(Global)},
		  columns/s/.style={column name=GPU(Shared)},
		]{\cdat}
\bigbreak		
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains size of matrix)},
			xlabel={Size of the matrix},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=red,mark=x]table {../data/dat_c.dat};
		\addlegendentry{CPU executing time}
		\addplot[smooth, color=blue,mark=x]table {../data/dat_gcu.dat};
		\addlegendentry{GPU(Global) executing time}
		\addplot[smooth, color=green,mark=x]table {../data/dat_scu.dat};
		\addlegendentry{GPU(Shared) executing time}
		\end{axis}
		\end{tikzpicture}

\section{GPU(Global/Shared) performance differences agains size of matrix}
		
		\pgfplotstablecreatecol[copy column from table={\scudat}{[index] 1}] {r} {\gcudat}

		\pgfplotstabletypeset[
		  columns/p/.style={column name=Matrix Size},
		  columns/q/.style={column name=GPU(Global)},
		  columns/r/.style={column name=GPU(Shared)},
		]{\gcudat}
\bigbreak		
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains size of matrix)},
			xlabel={Size of the matrix},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=blue,mark=x]table {../data/dat_gcu.dat};
		\addlegendentry{GPU(Global) executing time}
		\addplot[smooth, color=green,mark=x]table {../data/dat_scu.dat};
		\addlegendentry{GPU(Shared) executing time}
		\end{axis}
		\end{tikzpicture}

EOF2


cat > ../data/chapters/chapter02.tex << EOF3
\pagebreak
\pgfplotstableread{../data/dat_gxcu.dat}\gxcudat		
\pgfplotstableread{../data/dat_sxcu.dat}\sxcudat		

\section{GPU(Global) performance agains threads per block}
		
		\pgfplotstabletypeset[
		  columns/p/.style={column name=Threads per block},
		  columns/q/.style={column name=Executing time},
		]{\gxcudat}
\bigbreak		
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains threads per block)},
			xlabel={Threads per block},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=blue,mark=x]table {\gxcudat};
		\end{axis}
		\end{tikzpicture}

		 
\section{GPU(Shared) performance agains threads per block}
		
		\pgfplotstabletypeset[
		  columns/p/.style={column name=Threads per block},
		  columns/q/.style={column name=Executing time},
		]{\sxcudat}
\bigbreak		
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains threads per block)},
			xlabel={Threads per block},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=blue,mark=x]table {\sxcudat};
		\end{axis}
		\end{tikzpicture}

\section{GPU(Global/Shared) performance differences agains threads per block}
		
		\pgfplotstablecreatecol[copy column from table={\sxcudat}{[index] 1}] {r} {\gxcudat}

		\pgfplotstabletypeset[
		  columns/p/.style={column name=Threads per block},
		  columns/q/.style={column name=GPU(Global)},
		  columns/r/.style={column name=GPU(Shared)},
		]{\gxcudat}
\bigbreak		
		\begin{tikzpicture}
		\begin{axis}[
		title={Performence chart (Executing time agains threads per block)},
			xlabel={Threads per block},
			ylabel={Executing time [\si{milisecond}]},
			legend pos=north west,
			ymajorgrids=true,
			grid style=dashed,]
		\addplot[smooth, color=blue,mark=x]table {../data/dat_gxcu.dat};
		\addlegendentry{GPU(Global) executing time}
		\addplot[smooth, color=green,mark=x]table {../data/dat_sxcu.dat};
		\addlegendentry{GPU(Shared) executing time}
		\end{axis}
		\end{tikzpicture}

EOF3


cat > ../data/chapters/chapter03.tex << EOF4
\pagebreak
\section{Compare performance differences agains matrix size}
	The graph and tables represented in "Chapter 1" was calculated matrix multiplication average execute time agains matrix size. As user inputed data, every average execution time was calculated doing $itr iterations with constant block size (for gpu calculation) was $bsize. And user defined matrix sizes are $msizes.
\section{Compare performance differences agains threads per block}
	The graph and tables represented in "Chapter 2" was calculated matrix multiplication average execute time agains block size. As user inputed data, every average execution time was calculated doing $itr iterations with constant matrix size was $msize. And user defined block sizes are $bsizes.
EOF4

cat > ../data/chapters/chapter04.tex << EOF5
\pagebreak
\bigbreak
The CPU and GPU are two very different computing devices, and are meant to handle different types of computation. CPUs have fewer cores that can each handle more work per core and while the GPU has thousands of lightweight cores, making them good for smaller computations that need to be repeated often.
\bigbreak		
According to above graphs, You can performance calculation like matrix maltiplication with less time using GPU base programs, And data decomposition use for this matrix multiplication, so Shared memory is the best method for like these decompositions
EOF5

cat > ../data/chapters/chapter05.tex << EOF6
\pagebreak

\begin{thebibliography}{9}
 
\bibitem{cuda} 
Matrix Multiplication in CUDA
\\\https{://github.com/lzhengchun/matrix-cuda}

\bibitem{latex} 
ShareLaTex
\\\https{://www.sharelatex.com/}

\bibitem{shell} 
UNIX / LINUX Tutorial (TutorialsPoint)
\\\http{://www.tutorialspoint.com/unix/}
\end{thebibliography}

EOF6

cd ../data
pdflatex report.tex 
cp report.pdf ../
cd ..
reportName=`date '+%Y-%m-%d'`
mv report.pdf Report_$reportName.pdf
clear;
chmod 777 Report_$reportName.pdf
xdg-open Report_$reportName.pdf
sh run.sh

