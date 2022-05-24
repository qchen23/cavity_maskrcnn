
ckp=$1
t=$2
cat `find $ckp -name "cm-$2.txt"` > $ckp/all_cm.txt
cat $ckp/all_cm.txt | \
awk '{print $(NF-6), $(NF-3), $NF}' | \
sed 's/,//g' | \
awk '{p+=$1;r+=$2;f1+=$3}END{printf "precision = %lf, recall = %lf, f1_score = %lf\n", p/NR, r/NR, f1/NR}'
