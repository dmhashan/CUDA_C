cat > ../data/temp.tex << EOF1


\documentclass[12pt, letterpaper, twoside]{article}
\usepackage[utf8]{inputenc}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{textcomp}
\usepackage{siunitx}
\pgfplotsset{compat=1.12}

\title{CUDA Programming Assignment 01}
\author{UWU/CST/14/0014}
\date{\today}
 
\begin{document}
 
\begin{titlepage}
\maketitle
\end{titlepage}

\begin{tikzpicture}
\begin{axis}[
title={Performence chart (Executing time agains size of matrix)},
	xlabel={Size of the matrix},
	ylabel={Executing time [\si{milisecond}]},
	legend pos=north west,
	ymajorgrids=true,
	grid style=dashed,]
\addplot[color=red,mark=x]table {../data/dat_c.dat};
\addlegendentry{CPU executing time}
\addplot[color=blue,mark=x]table {../data/dat_gcu.dat};
\addlegendentry{GPU(Global) executing time}
\addplot[color=green,mark=x]table {../data/dat_scu.dat};
\addlegendentry{GPU(Shared) executing time}
\end{axis}
\end{tikzpicture}


\begin{tikzpicture}
\begin{axis}[
title={Performence chart (Executing time agains size of matrix)},
	xlabel={Size of the matrix},
	ylabel={Executing time [\si{milisecond}]},
	legend pos=north west,
	ymajorgrids=true,
	grid style=dashed,]
\addplot[color=blue,mark=x]table {../data/dat_gcu.dat};
\addlegendentry{GPU(Global) executing time}
\addplot[color=green,mark=x]table {../data/dat_scu.dat};
\addlegendentry{GPU(Shared) executing time}
\end{axis}
\end{tikzpicture}


\begin{tikzpicture}
\begin{axis}[
title={Performence chart (Excuting time agains threads per block)},
	xlabel={Threads per block},
	ylabel={Executing time [\si{milisecond}]},
	legend pos=north west,
	ymajorgrids=true,
	grid style=dashed,]
\addplot[color=red,mark=x]table {../data/dat_gcu.dat};
\addlegendentry{GPU(Global) executing time}
\addplot[color=blue,mark=x]table {../data/dat_scu.dat};
\addlegendentry{GPU(Shared) executing time}
\end{axis}
\end{tikzpicture}



\end{document}


EOF1

cd ../data
pdflatex temp.tex 
cp temp.pdf ../
