# Credits to Jeff Amelang, 2013

import os

for filename in os.listdir("output"):
  if (os.path.splitext(filename)[1] == ".dat"):
    file = open('output/gnuplotScript.gnuplot', 'w')
    file.write('set datafile separator ","\n')
    file.write('plot "output/%s" using 1:2 with lines lw 2 lc 1 lt 1 title "displacements"\n' % filename)
    file.write('set grid\n')
    file.write('set xrange [0:1]\n')
    file.write('set yrange [-2:2]\n')
    file.write('set terminal pngcairo dashed size 800,600\n')
    file.write('set output "output/%s.png"\n' % os.path.splitext(filename)[0])
    file.write('set key right box\n')
    file.write('set ylabel "Displacement"\n')
    file.write('set xlabel "Position"\n')
    file.write('replot\n')
    file.close()
    os.system('gnuplot output/gnuplotScript.gnuplot')
  else:
    print "not processing file %s, it doesn't appear to be a data file\n" % filename
