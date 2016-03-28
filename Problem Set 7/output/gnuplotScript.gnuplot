set datafile separator ","
plot "output/GPU_data_00033332.dat" using 1:2 with lines lw 2 lc 1 lt 1 title "displacements"
set grid
set xrange [0:1]
set yrange [-2:2]
set terminal pngcairo dashed size 800,600
set output "output/GPU_data_00033332.png"
set key right box
set ylabel "Displacement"
set xlabel "Position"
replot
