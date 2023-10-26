
# set terminal pngcairo enhanced color font 'Helvetica,10'
# set output '4pep.png'

maxopt=` cat 4pep_opt..csv|sort -n -k2|tail -n 1|cut -f2 `
maxdpt=` cat 16pep0830f0000.dpt|sort -n -k2|tail -n 1|cut -c 25- `
scale2=(maxopt/maxdpt)

print maxopt
print maxdpt, scale2

set encoding utf8
set title "4PEP"
set yrange [ 0 : 0.03 ]
set xrange [ 1720 : 1580 ]
set xlabel "wave number / cm^-^1"
set ylabel "absorption"
set key

plot "4pep_opt..csv" u 1:2 t sprintf("optimized data | Karjalainen, Ersmark, Barth 2012") w l ls 1 lw 5 lc "red",\
     "16pep0830f0000.dpt" u 1:($2*scale2) t sprintf("experimental data, scale factor %3.4f | Oberg, Ruysschaert, Goormaghtigh 2003", scale2) w l ls 1 lw 5 lc "blue"
pause -1